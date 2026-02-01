import json
import os
import re
import boto3
from botocore.exceptions import ClientError

# ---- Config ----
REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# Input S3 (reviews)
IN_BUCKET = "app-review-mock-data"
IN_KEY = "reviews.json"

# Input S3 (activity logs)
ACT_BUCKET = "activity-mock-data"
ACT_KEY = "activity_log.json"

# Input S3 (problem tickets)  <-- NEW
TICKETS_BUCKET = "problem-tickets-mock-data"
TICKETS_KEY = "problem_tickets.json"

# Output S3 (feature spec)
OUT_BUCKET = "app-features-spec"
OUT_KEY = "feature_spec.json"

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

ALLOWED_SEVERITIES = {"low", "medium", "high"}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

# Tickets priority → severity mapping (used as guidance in prompt + sanity checks)
TICKET_PRIORITY_TO_SEVERITY = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "high"
}


# -------------------- S3 helpers --------------------
def _load_json_from_s3(bucket: str, key: str):
    resp = s3.get_object(Bucket=bucket, Key=key)
    raw = resp["Body"].read().decode("utf-8")
    return json.loads(raw)


def _load_reviews_from_s3(bucket: str, key: str) -> list:
    """
    Accepts either:
      - New format: top-level JSON list of review dicts
      - Old format: {"reviews":[...]}
    """
    payload = _load_json_from_s3(bucket, key)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
        return payload["reviews"]

    raise ValueError("S3 JSON must be either a list of reviews or an object with a 'reviews' list.")


def _load_activity_from_s3(bucket: str, key: str) -> list:
    """
    Accepts either:
      - New format: top-level JSON list of event dicts
      - Old format: {"events":[...]}
    """
    payload = _load_json_from_s3(bucket, key)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return payload["events"]

    raise ValueError("Activity log JSON must be either a list of events or an object with an 'events' list.")


def _load_tickets_from_s3(bucket: str, key: str) -> list:
    """
    Accepts either:
      - Top-level JSON list of ticket dicts
      - Old format: {"tickets":[...]}
    """
    payload = _load_json_from_s3(bucket, key)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("tickets"), list):
        return payload["tickets"]

    raise ValueError("Problem tickets JSON must be either a list of tickets or an object with a 'tickets' list.")


def _write_json_to_s3(bucket: str, key: str, obj):
    body = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


# -------------------- Output parsing/normalization --------------------
def _extract_first_json_array(text: str):
    """
    Extract and parse the first JSON array from a model response.
    """
    t = (text or "").strip()

    # direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # best-effort: find first [...] block
    m = re.search(r"\[[\s\S]*\]", t)
    if not m:
        raise ValueError("Model output did not contain a JSON array.")

    block = m.group(0)

    # remove trailing commas before ] or }
    block = re.sub(r",\s*([\]\}])", r"\1", block)

    # if single quotes used everywhere (rare), attempt conversion
    if "'" in block and '"' not in block:
        block = block.replace("'", '"')

    obj = json.loads(block)
    if not isinstance(obj, list):
        raise ValueError("Parsed JSON is not an array.")
    return obj


def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _normalize_spec_with_use_count(spec_list: list) -> list:
    """
    Enforce EXACT schema:
    [
      {
        "category": <str>,
        "keywords": [<str>, ...],
        "possible_issues": [<str>, ...],
        "severity": "low"|"medium"|"high",
        "use_count": <int>
      },
      ...
    ]
    """
    if not isinstance(spec_list, list):
        return []

    cleaned = []
    for item in spec_list:
        if not isinstance(item, dict):
            continue

        category = item.get("category", "")
        if not isinstance(category, str):
            category = str(category) if category is not None else ""
        category = category.strip()

        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]

        possible_issues = item.get("possible_issues", [])
        if not isinstance(possible_issues, list):
            possible_issues = []
        possible_issues = [p.strip() for p in possible_issues if isinstance(p, str) and p.strip()]

        severity = item.get("severity", "medium")
        if not isinstance(severity, str):
            severity = "medium"
        severity = severity.strip().lower()
        if severity not in ALLOWED_SEVERITIES:
            severity = "medium"

        use_count = item.get("use_count", 0)
        try:
            use_count = int(use_count)
            if use_count < 0:
                use_count = 0
        except Exception:
            use_count = 0

        keywords = _dedupe_preserve_order(keywords)
        possible_issues = _dedupe_preserve_order(possible_issues)

        if category:
            cleaned.append({
                "category": category,
                "keywords": keywords,
                "possible_issues": possible_issues,
                "severity": severity,
                "use_count": use_count
            })

    return cleaned


def _collapse_duplicate_categories_with_use_count(spec: list) -> list:
    """
    Merge duplicates:
      - keywords & issues union
      - severity = max (high > medium > low)
      - use_count = sum
    """
    merged = {}
    order = []

    for item in spec:
        cat = item["category"].strip()
        key = cat.lower()
        if key not in merged:
            merged[key] = {
                "category": cat,
                "keywords": [],
                "possible_issues": [],
                "severity": item["severity"],
                "use_count": 0,
            }
            order.append(key)

        merged[key]["keywords"].extend(item.get("keywords", []))
        merged[key]["possible_issues"].extend(item.get("possible_issues", []))

        merged[key]["use_count"] += int(item.get("use_count", 0) or 0)

        cur = merged[key]["severity"]
        inc = item["severity"]
        if SEVERITY_RANK.get(inc, 1) > SEVERITY_RANK.get(cur, 1):
            merged[key]["severity"] = inc

    out = []
    for key in order:
        merged[key]["keywords"] = _dedupe_preserve_order(
            [k for k in merged[key]["keywords"] if isinstance(k, str) and k.strip()]
        )
        merged[key]["possible_issues"] = _dedupe_preserve_order(
            [p for p in merged[key]["possible_issues"] if isinstance(p, str) and p.strip()]
        )
        out.append(merged[key])

    return out


# -------------------- Prompt --------------------
def _build_batch_prompt(reviews: list, activity_events: list, tickets: list) -> str:
    """
    Instruct Claude to output EXACTLY the requested JSON array format,
    adding use_count derived from activity logs, and using tickets as extra severity signal.
    """
    # Slim reviews
    slim_reviews = []
    for r in reviews:
        if not isinstance(r, dict):
            continue
        slim_reviews.append({
            "review_id": r.get("review_id"),
            "title": r.get("title"),
            "review": r.get("review"),
            "rating": r.get("rating"),
            "timestamp": r.get("timestamp"),
        })

    # Slim activity events (keep small)
    slim_events = []
    for e in activity_events:
        if not isinstance(e, dict):
            continue
        slim_events.append({
            "timestamp": e.get("timestamp"),
            "user_id": e.get("user_id"),
            "session_id": e.get("session_id"),
            "event_type": e.get("event_type"),
            "feature": e.get("feature"),
        })

    # Slim tickets
    slim_tickets = []
    for t in tickets:
        if not isinstance(t, dict):
            continue
        cust = t.get("customer") if isinstance(t.get("customer"), dict) else {}
        slim_tickets.append({
            "ticket_id": t.get("ticket_id"),
            "customer_name": cust.get("name"),
            "subject": t.get("subject"),
            "description": t.get("description"),
            "status": t.get("status"),
            "priority": t.get("priority"),
            "created_at": t.get("created_at"),
            "updated_at": t.get("updated_at"),
        })

    reviews_json = json.dumps(slim_reviews, ensure_ascii=False)
    events_json = json.dumps(slim_events, ensure_ascii=False)
    tickets_json = json.dumps(slim_tickets, ensure_ascii=False)

    return f"""
You are given:
1) User reviews
2) User activity events
3) Problem tickets (support tickets)

Produce a feature-spec JSON array in the EXACT format below.
Return ONLY a valid JSON array, nothing else.

Required output format (EXACT keys and types):
[
  {{
    "category": "<Title Case category name>",
    "keywords": ["<keyword1>", "<keyword2>", ...],
    "possible_issues": ["<issue sentence 1>", "<issue sentence 2>", ...],
    "severity": "low" | "medium" | "high",
    "use_count": 0
  }},
  ...
]

CRITICAL category aggregation:
- Output EXACTLY ONE object per category (no duplicates).
- Determine ONE overall severity per category by analyzing ALL reviews AND tickets that fit that category.
  Use frequency + rating distribution + language strength + ticket priority.
- Determine use_count per category by counting relevant activity events.

SEVERITY guidance:
- high: repeated and/or severe complaints OR ratings <= 2 OR critical/high priority tickets
- medium: mixed/important but not total blocker OR rating == 3 OR medium priority tickets
- low: minor/rare OR mostly ratings >= 4 OR low priority tickets

Ticket priority mapping hint:
- critical/high -> high severity signal
- medium -> medium severity signal
- low -> low severity signal

COUNTING RULES for use_count:
- Count a "use" if event_type is one of:
  "feature_open", "summary_generated", "insights_viewed", "recommendations_loaded"
- Map events into categories using event.feature:
  * "Summaries": event.feature in ["email_summary", "meeting_summary"]
  * "Insights": event.feature == "daily_insights"
  * "Recommendations": event.feature == "recommendations"
  * "Accuracy": event_type in ["open_source_item", "context_expand"] (feature can be any)
- use_count is the TOTAL number of qualifying events for that category.
- use_count must be an integer.

Category inference:
- Categories should be high-level feature areas inferred from reviews+tickets.
- When tickets mention missing context/details in answers/summaries, that should reinforce "Summaries" and/or "Accuracy".
- When tickets mention generic recommendations, that should reinforce "Recommendations".

Rules:
- "keywords" are short, lowercase terms users might mention.
- "possible_issues" are concrete issues derived from the review/ticket text (not generic).
- Deduplicate keywords/issues.

Input reviews JSON:
{reviews_json}

Input activity events JSON:
{events_json}

Input problem tickets JSON:
{tickets_json}
""".strip()


def lambda_handler(event, context):
    if not MODEL_ID:
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "Missing BEDROCK_MODEL_ID (env var or fallback)."})
        }

    # 1) Load reviews from S3
    try:
        reviews = _load_reviews_from_s3(IN_BUCKET, IN_KEY)
    except Exception as e:
        return {
            "statusCode": 502,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": "Failed to load reviews from S3",
                "bucket": IN_BUCKET,
                "key": IN_KEY,
                "details": str(e),
            })
        }

    # 2) Load activity logs from S3
    try:
        activity_events = _load_activity_from_s3(ACT_BUCKET, ACT_KEY)
    except Exception as e:
        return {
            "statusCode": 502,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": "Failed to load activity log from S3",
                "bucket": ACT_BUCKET,
                "key": ACT_KEY,
                "details": str(e),
            })
        }

    # 3) Load problem tickets from S3  <-- NEW
    try:
        tickets = _load_tickets_from_s3(TICKETS_BUCKET, TICKETS_KEY)
    except Exception as e:
        return {
            "statusCode": 502,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": "Failed to load problem tickets from S3",
                "bucket": TICKETS_BUCKET,
                "key": TICKETS_KEY,
                "details": str(e),
            })
        }

    # 4) Call Bedrock once (batch)
    prompt = _build_batch_prompt(reviews, activity_events, tickets)

    try:
        resp = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": "You output strictly formatted JSON following the given schema."}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1300, "temperature": 0.0},
        )
    except (ClientError, Exception) as e:
        return {
            "statusCode": 502,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": f"Bedrock invoke failed: {str(e)}"})
        }

    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    text_out = "".join(b.get("text", "") for b in blocks).strip()

    # 5) Parse + normalize
    try:
        spec_list = _extract_first_json_array(text_out)
    except Exception as e:
        return {
            "statusCode": 502,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": "Model did not return a valid JSON array in the required format.",
                "details": str(e),
                "model_output": text_out[:2000]
            })
        }

    normalized = _collapse_duplicate_categories_with_use_count(
        _normalize_spec_with_use_count(spec_list)
    )

    # 6) Write output (top-level list) to S3
    try:
        _write_json_to_s3(OUT_BUCKET, OUT_KEY, normalized)
    except Exception as e:
        return {
            "statusCode": 502,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": "Failed to write feature spec to S3",
                "bucket": OUT_BUCKET,
                "key": OUT_KEY,
                "details": str(e),
            })
        }

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({
            "message": "Wrote feature spec (with use_count; tickets-informed severity) to S3",
            "output_bucket": OUT_BUCKET,
            "output_key": OUT_KEY,
            "count": len(normalized),
            "model_id": MODEL_ID
        })
    }