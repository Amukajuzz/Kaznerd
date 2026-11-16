"""Analytical SQL queries over the KazNERD database.

Each entry: key -> (title, SQL). The techniques used are noted in the title.
"""

BEST_MODEL = "(SELECT model_id FROM model_scores ORDER BY f1_micro DESC LIMIT 1)"

QUERIES = {
    "q01_entity_counts": (
        "Mentions, unique surface forms and mean length per class [JOIN, GROUP BY]",
        """
        SELECT e.type_code, t.description,
               COUNT(*)                   AS mentions,
               COUNT(DISTINCT e.text)     AS unique_forms,
               ROUND(AVG(e.n_tokens), 2)  AS avg_tokens
        FROM entities e
        JOIN entity_types t USING (type_code)
        GROUP BY e.type_code, t.description
        ORDER BY mentions DESC;
        """),
    "q02_frequent_classes_by_split": (
        "Classes with at least 500 mentions in a split [JOIN, GROUP BY, HAVING]",
        """
        SELECT s.split, e.type_code, COUNT(*) AS mentions
        FROM entities e
        JOIN sentences s USING (sentence_id)
        GROUP BY s.split, e.type_code
        HAVING COUNT(*) >= 500
        ORDER BY s.split, mentions DESC;
        """),
    "q03_top_entities_per_class": (
        "Top-5 most frequent entities for key classes [CTE, ROW_NUMBER window]",
        """
        WITH counts AS (
            SELECT type_code, text, COUNT(*) AS n
            FROM entities
            GROUP BY type_code, text
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY type_code ORDER BY n DESC, text) AS rn
            FROM counts
        )
        SELECT type_code, rn AS rank, text, n
        FROM ranked
        WHERE rn <= 5 AND type_code IN ('PERSON', 'GPE', 'ORGANISATION', 'LOCATION', 'POSITION')
        ORDER BY type_code, rn;
        """),
    "q04_class_pareto": (
        "Share and cumulative share of mentions per class [CTE, SUM() OVER window]",
        """
        WITH c AS (SELECT type_code, COUNT(*) AS n FROM entities GROUP BY type_code)
        SELECT type_code, n,
               ROUND(100.0 * n / SUM(n) OVER (), 2) AS pct,
               ROUND(100.0 * SUM(n) OVER (ORDER BY n DESC ROWS UNBOUNDED PRECEDING)
                     / SUM(n) OVER (), 2) AS cumulative_pct
        FROM c
        ORDER BY n DESC;
        """),
    "q05_split_statistics": (
        "Sentence length and entity density per split [CTE, LEFT JOIN, aggregation]",
        """
        WITH per_sentence AS (
            SELECT s.sentence_id, s.split, s.n_tokens, COUNT(e.entity_id) AS n_entities
            FROM sentences s
            LEFT JOIN entities e USING (sentence_id)
            GROUP BY s.sentence_id, s.split, s.n_tokens
        )
        SELECT split,
               COUNT(*)                                        AS sentences,
               SUM(n_tokens)                                   AS tokens,
               ROUND(AVG(n_tokens), 2)                         AS avg_tokens,
               ROUND(AVG(n_entities), 3)                       AS avg_entities,
               ROUND(100.0 * SUM(n_entities = 0) / COUNT(*), 1) AS pct_without_entities
        FROM per_sentence
        GROUP BY split
        ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'valid' THEN 2 ELSE 3 END;
        """),
    "q06_entity_dense_sentences": (
        "Sentences with > 3x the average number of entities [nested subquery, HAVING]",
        """
        SELECT s.sentence_id, s.split, s.n_tokens, COUNT(*) AS n_entities, s.text
        FROM sentences s
        JOIN entities e USING (sentence_id)
        GROUP BY s.sentence_id
        HAVING COUNT(*) > (SELECT 3 * AVG(cnt)
                           FROM (SELECT COUNT(*) AS cnt FROM entities GROUP BY sentence_id))
        ORDER BY n_entities DESC
        LIMIT 10;
        """),
    "q07_class_cooccurrence": (
        "Which classes appear together in one sentence [self-JOIN, GROUP BY]",
        """
        SELECT a.type_code AS type_a, b.type_code AS type_b,
               COUNT(DISTINCT a.sentence_id) AS sentences
        FROM entities a
        JOIN entities b ON a.sentence_id = b.sentence_id AND a.type_code < b.type_code
        GROUP BY a.type_code, b.type_code
        ORDER BY sentences DESC
        LIMIT 15;
        """),
    "q08_normalisation_by_class": (
        "Stop-word share and vocabulary of forms / stems / lemmas per class [CASE, COUNT DISTINCT]",
        """
        SELECT CASE WHEN tag = 'O' THEN 'O' ELSE substr(tag, 3) END AS class,
               COUNT(*)                          AS tokens,
               ROUND(100.0 * AVG(is_stopword), 2) AS pct_stopwords,
               COUNT(DISTINCT lower)             AS forms,
               COUNT(DISTINCT stem)              AS stems,
               COUNT(DISTINCT lemma)             AS lemmas
        FROM tokens
        GROUP BY class
        ORDER BY tokens DESC;
        """),
    "q09_model_leaderboard": (
        "Model ranking with gain over the next-weaker model [JOIN, RANK and LAG windows]",
        """
        SELECT m.name, s.f1_micro, s.f1_macro, s.accuracy,
               m.train_time_s, m.peak_mem_mb,
               RANK() OVER (ORDER BY s.f1_micro DESC) AS rank_f1,
               ROUND(s.f1_micro - LAG(s.f1_micro) OVER (ORDER BY s.f1_micro), 4) AS gain_vs_previous
        FROM model_scores s
        JOIN models m USING (model_id)
        ORDER BY s.f1_micro DESC;
        """),
    "q10_hardest_classes_best_model": (
        "Ten weakest classes of the best model [scalar subquery]",
        f"""
        SELECT c.type_code, ROUND(c.precision, 3) AS precision, ROUND(c.recall, 3) AS recall,
               ROUND(c.f1, 3) AS f1, c.support
        FROM class_scores c
        WHERE c.model_id = {BEST_MODEL}
        ORDER BY c.f1
        LIMIT 10;
        """),
    "q11_error_profile": (
        "Error-type distribution per model [JOIN, GROUP BY, SUM() OVER PARTITION]",
        """
        SELECT m.name, e.error_type, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY m.name), 1) AS pct
        FROM errors e
        JOIN models m USING (model_id)
        GROUP BY m.name, e.error_type
        ORDER BY m.name, n DESC;
        """),
    "q12_top_confusions_best_model": (
        "Most frequent gold -> predicted class confusions of the best model [subquery, GROUP BY]",
        f"""
        SELECT gold_type, pred_type, COUNT(*) AS n
        FROM errors
        WHERE error_type IN ('type_error', 'type_and_boundary') AND model_id = {BEST_MODEL}
        GROUP BY gold_type, pred_type
        ORDER BY n DESC
        LIMIT 10;
        """),
    "q13_seen_vs_unseen_recall": (
        "Recall on test entities seen vs unseen in train [CTE, EXISTS subquery, LEFT JOIN]",
        """
        WITH train_entities AS (
            SELECT DISTINCT e.type_code, e.text
            FROM entities e JOIN sentences s USING (sentence_id)
            WHERE s.split = 'train'
        ), test_gold AS (
            SELECT e.*,
                   CASE WHEN EXISTS (SELECT 1 FROM train_entities t
                                     WHERE t.type_code = e.type_code AND t.text = e.text)
                        THEN 'seen' ELSE 'unseen' END AS novelty
            FROM entities e JOIN sentences s USING (sentence_id)
            WHERE s.split = 'test'
        )
        SELECT m.name, g.novelty, COUNT(*) AS gold,
               SUM(p.pred_id IS NOT NULL) AS found,
               ROUND(100.0 * SUM(p.pred_id IS NOT NULL) / COUNT(*), 2) AS recall_pct
        FROM test_gold g
        CROSS JOIN models m
        LEFT JOIN predicted_entities p
               ON p.model_id = m.model_id AND p.sentence_id = g.sentence_id
              AND p.start_pos = g.start_pos AND p.end_pos = g.end_pos
              AND p.type_code = g.type_code
        GROUP BY m.name, g.novelty
        ORDER BY m.name, g.novelty;
        """),
    "q14_context_keywords": (
        "Top context keywords signalling each class [subquery, GROUP_CONCAT]",
        """
        SELECT type_code, GROUP_CONCAT(term, ', ') AS keywords
        FROM (SELECT type_code, term FROM keywords WHERE rank <= 8 ORDER BY type_code, rank)
        GROUP BY type_code
        ORDER BY type_code;
        """),
}
