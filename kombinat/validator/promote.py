import asyncpg


async def maybe_promote_pair(pool: asyncpg.Pool, pair_id: object) -> bool:
    """Promote pair to verified/rejected if enough annotations exist. Returns True if promoted.

    Honeypot pairs are never promoted: their status must stay 'unlabeled' so they
    remain claimable indefinitely for quality control. Promoting one would remove
    it from the honeypot pool after `required_annotations` uses.
    """
    row = await pool.fetchrow(
        """SELECT p.required_annotations, (h.pair_id IS NOT NULL) AS is_honeypot
        FROM pairs p
        LEFT JOIN honeypots h ON h.pair_id = p.id
        WHERE p.id = $1""",
        pair_id,
    )
    if row is None or row["is_honeypot"]:
        return False

    required = int(row["required_annotations"])
    annotations = await pool.fetch("SELECT label FROM annotations WHERE pair_id = $1", pair_id)

    if len(annotations) < required:
        return False

    # Majority vote on the binary relevance decision: labels 0-1 count as
    # "not relevant", 2-3 as "relevant". Voting on the exact 0-3 label would
    # reject adjacent-grade agreement like [2,3] — the most common outcome for
    # LLM judges on genuinely relevant pairs — so nearly every hard pair would
    # end up 'rejected'. The exact labels stay in the annotations table.
    labels = [int(a["label"]) for a in annotations]
    relevant_votes = sum(1 for label in labels if label >= 2)
    agreement_count = max(relevant_votes, len(labels) - relevant_votes)

    if agreement_count > len(labels) // 2:
        await pool.execute("UPDATE pairs SET status = 'verified' WHERE id = $1", pair_id)
    else:
        await pool.execute("UPDATE pairs SET status = 'rejected' WHERE id = $1", pair_id)

    return True
