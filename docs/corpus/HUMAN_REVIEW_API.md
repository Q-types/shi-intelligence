# Human Review API

## Overview

The Human Review API provides REST endpoints for the human-in-the-loop labelling workflow. It enables reviewers to label items, verify labels, dispute disagreements, and track quality metrics.

## Base URL

```
/api/v1/review
```

## Authentication

Authentication is handled at the application level. All endpoints require a valid session.

---

## Endpoints

### GET /queue

Get the prioritized review queue.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter` | string | `pending` | Queue filter: `pending`, `labelled`, `disputed`, `verification` |
| `domain` | string | null | Filter by domain (e.g., `exit_event`) |
| `limit` | int | 50 | Maximum items (1-200) |

**Response:**

```json
[
  {
    "label_id": "lbl_123",
    "domain": "exit_event",
    "object_type": "exit",
    "object_id": "exit_456",
    "proposed_label": "dex_sell",
    "model_confidence": 0.85,
    "priority_score": 0.72,
    "priority_factors": ["low_confidence:0.65", "high_impact:exit_event"],
    "created_at": "2024-01-15T10:30:00Z",
    "evidence_summary": "DEX sell of 1000 tokens via Raydium"
  }
]
```

**Priority Factors:**
- `low_confidence:{value}` - Model confidence below threshold
- `high_impact:{domain}` - High-impact domain
- `rare_class:{label}` - Underrepresented class
- `model_human_disagree` - Previous model-human disagreement

---

### GET /item/{label_id}

Get detailed information about a label including evidence.

**Response:**

```json
{
  "label_id": "lbl_123",
  "domain": "exit_event",
  "object_type": "exit",
  "object_id": "exit_456",
  "proposed_label": "dex_sell",
  "human_label": null,
  "final_label": "dex_sell",
  "model_confidence": 0.85,
  "human_confidence": null,
  "review_status": "pending",
  "source_model": "exit_classifier",
  "source_model_version": "1.0.0",
  "data_version": "2024-01-15",
  "evidence": {
    "exit_id": "exit_456",
    "wallet": "abc...",
    "token_mint": "xyz...",
    "amount_tokens": 1000.0,
    "value_sol": 5.5,
    "counterparty": "raydium_pool",
    "transaction_signature": "tx...",
    "program_id": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "confidence_factors": {
      "signature_match": 0.95
    }
  },
  "created_at": "2024-01-15T10:30:00Z",
  "reviewed_at": null,
  "reviewer_id": null,
  "notes": null,
  "version_history": []
}
```

---

### POST /item/{label_id}/label

Submit a human label for an item.

**Request Body:**

```json
{
  "human_label": "dex_sell",
  "human_confidence": 0.95,
  "reviewer_id": "reviewer_001",
  "notes": "Clear Raydium swap signature"
}
```

**Validation:**
- `human_label`: Required, non-empty string
- `human_confidence`: Required, float 0.0-1.0
- `reviewer_id`: Required, non-empty string
- `notes`: Optional string

**Response:**

```json
{
  "success": true,
  "label_id": "lbl_123",
  "new_status": "labelled",
  "message": "Label submitted: dex_sell"
}
```

**Errors:**
- `404`: Label not found
- `400`: Label not in PENDING status

---

### POST /item/{label_id}/verify

Verify or dispute a labelled item (second review).

**Request Body:**

```json
{
  "verified": true,
  "second_label": null,
  "reviewer_id": "reviewer_002",
  "notes": "Confirmed, signature matches"
}
```

For disputes:

```json
{
  "verified": false,
  "second_label": "transfer_out",
  "reviewer_id": "reviewer_002",
  "notes": "This looks like a transfer, not a sell"
}
```

**Response:**

```json
{
  "success": true,
  "label_id": "lbl_123",
  "new_status": "verified",
  "message": "Label verified"
}
```

Or for disputes:

```json
{
  "success": true,
  "label_id": "lbl_123",
  "new_status": "disputed",
  "message": "Label disputed: dex_sell vs transfer_out"
}
```

**Errors:**
- `404`: Label not found
- `400`: Label not in LABELLED status
- `400`: Must provide different `second_label` when `verified=false`

---

### POST /item/{label_id}/dispute

Dispute an existing label (can be used on LABELLED or VERIFIED items).

**Request Body:**

```json
{
  "disputed_label": "transfer_out",
  "reason": "Upon closer inspection, this is a transfer to another wallet, not a DEX sell",
  "reviewer_id": "reviewer_003"
}
```

**Response:**

```json
{
  "success": true,
  "label_id": "lbl_123",
  "new_status": "disputed",
  "message": "Label disputed: Upon closer inspection..."
}
```

**Errors:**
- `404`: Label not found
- `400`: Cannot dispute labels with status other than LABELLED or VERIFIED

---

### GET /progress

Get review progress statistics.

**Response:**

```json
{
  "total_labels": 1500,
  "pending": 800,
  "labelled": 400,
  "verified": 250,
  "disputed": 50,
  "completion_rate": 0.433,
  "verification_rate": 0.385,
  "dispute_rate": 0.033,
  "by_domain": {
    "exit_event": {
      "pending": 300,
      "labelled": 200,
      "verified": 150,
      "disputed": 20
    },
    "coordination": {
      "pending": 200,
      "labelled": 100,
      "verified": 50,
      "disputed": 10
    }
  }
}
```

---

### GET /metrics

Get label quality metrics.

**Response:**

```json
{
  "computed_at": "2024-01-15T12:00:00Z",
  "total_labels": 1500,
  "total_reviewed": 700,
  "total_verified": 250,
  "overall_inter_reviewer_agreement": 0.87,
  "overall_cohens_kappa": 0.72,
  "overall_model_human_agreement": 0.81,
  "kappa_threshold_met": true,
  "agreement_threshold_met": true,
  "ready_for_training": true,
  "recommendations": [
    "Label quality metrics look good. Ready for model training."
  ],
  "domain_metrics": {
    "exit_event": {
      "domain": "exit_event",
      "total_labels": 670,
      "pending": 300,
      "labelled": 200,
      "verified": 150,
      "disputed": 20,
      "inter_reviewer_agreement": 0.89,
      "cohens_kappa": 0.75,
      "model_human_agreement": 0.83,
      "label_distribution": {
        "dex_sell": 450,
        "transfer_out": 120,
        "lp_add": 50,
        "unknown": 50
      },
      "ambiguous_rate": 0.02,
      "needs_context_rate": 0.01,
      "disagreement_rate": 0.03
    }
  }
}
```

---

### GET /domains

Get available domains and their valid labels.

**Response:**

```json
{
  "exit_event": [
    "dex_sell",
    "transfer_out",
    "lp_add",
    "lp_remove",
    "cex_deposit",
    "burn",
    "swap_intermediate",
    "stake",
    "bridge_out",
    "unknown",
    "ambiguous",
    "needs_more_context"
  ],
  "coordination": [
    "true_coordinated",
    "false_positive",
    "partially_coordinated",
    "unknown_coordination",
    "legitimate_coordination"
  ]
}
```

---

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid input or state) |
| 404 | Label not found |
| 500 | Internal server error |

---

## Review Workflow

```
1. GET /queue?filter=pending
   → Get items to review

2. GET /item/{label_id}
   → View evidence and details

3. POST /item/{label_id}/label
   → Submit label (PENDING → LABELLED)

4. GET /queue?filter=verification
   → Get items for second review

5. POST /item/{label_id}/verify
   → Verify (LABELLED → VERIFIED)
   or
   → Dispute (LABELLED → DISPUTED)

6. GET /queue?filter=disputed
   → Get disputed items for resolution

7. GET /metrics
   → Check quality metrics
```

---

## Integration Example

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000/api/v1/review")

# Get pending queue
queue = client.get("/queue", params={"filter": "pending", "limit": 10}).json()

for item in queue:
    # Get full details
    detail = client.get(f"/item/{item['label_id']}").json()

    # Review evidence and submit label
    response = client.post(
        f"/item/{item['label_id']}/label",
        json={
            "human_label": "dex_sell",
            "human_confidence": 0.95,
            "reviewer_id": "reviewer_001",
            "notes": "Clear swap signature",
        },
    )
    print(response.json())
```
