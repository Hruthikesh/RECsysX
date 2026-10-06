-- Queries used by scripts/db_check.py. Each block starts with "-- name: <check name>".
-- The script runs them and compares the result with the same number computed in pandas.

-- name: n_users
SELECT count(*) FROM users;

-- name: n_items
SELECT count(*) FROM items;

-- name: n_ratings
SELECT count(*) FROM ratings;

-- name: n_train_positives
SELECT count(*) FROM train_positives;

-- name: n_eval_positives_val_test_users
SELECT count(*)
FROM ratings r JOIN users u USING (user_idx)
WHERE r.period = 'eval' AND r.rating >= 4 AND u.eval_role IN ('val', 'test');

-- name: matrix_density
SELECT count(*)::float / ((SELECT count(*) FROM users) * (SELECT count(*) FROM items)) FROM ratings;

-- name: latest_train_rating_before_first_eval
-- the split is only valid if every training rating is older than every eval rating
SELECT (SELECT max(rated_at) FROM ratings WHERE period = 'train')
     < (SELECT min(rated_at) FROM ratings WHERE period = 'eval');

-- name: cold_test_users_with_train_positives
SELECT count(*) FROM users u
WHERE u.eval_role = 'test' AND u.user_group = 'cold'
  AND EXISTS (SELECT 1 FROM train_positives p WHERE p.user_idx = u.user_idx);

-- name: top_item_by_train_positives
SELECT item_idx FROM train_positives GROUP BY item_idx ORDER BY count(*) DESC, item_idx LIMIT 1;

-- name: share_of_positives_on_head_items
SELECT avg(CASE WHEN i.is_head THEN 1.0 ELSE 0.0 END)
FROM train_positives p JOIN items i USING (item_idx);

-- name: median_user_train_ratings
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY n)
FROM (SELECT count(*) AS n FROM ratings WHERE period = 'train' GROUP BY user_idx) t;

-- name: drama_share_of_items
SELECT avg(CASE WHEN 'Drama' = ANY (genres) THEN 1.0 ELSE 0.0 END) FROM items;
