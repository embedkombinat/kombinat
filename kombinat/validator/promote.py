import asyncpg


async def maybe_promote_pair(pool: asyncpg.Pool, pair_id: object) -> bool:
    """Promote pair to verified/rejected if enough annotations exist. Returns True if promoted."""
    row = await pool.fetchrow("SELECT required_annotations FROM pairs WHERE id = $1", pair_id)
    if row is None:
        return False

    required = int(row["required_annotations"])
    annotations = await pool.fetch("SELECT label FROM annotations WHERE pair_id = $1", pair_id)

    if len(annotations) < required:
        return False

    # Majority vote
    labels = [int(a["label"]) for a in annotations]
    majority_label = max(set(labels), key=labels.count)
    agreement_count = labels.count(majority_label)

    if agreement_count > len(labels) // 2:
        await pool.execute("UPDATE pairs SET status = 'verified' WHERE id = $1", pair_id)
    else:
        await pool.execute("UPDATE pairs SET status = 'rejected' WHERE id = $1", pair_id)

    return True
