# 5 daily-work tasks x 4 items = 20 prompts. All inputs are synthetic text written for this experiment (no private data).
# Auto-scored tasks carry a gold answer / test cases; summary and tweet are judged by Grok (1-5) in one batched call.
TRIAGE_LABELS = ["AI research", "Job alert", "Promotion", "Product update"]
TRIAGE = [
    ("Weekly Papers Digest: 'Sparse attention scales to 1M tokens'", "This week: a new paper shows sparse attention matching dense quality at 1M context, plus two RL fine-tuning results.", "AI research"),
    ("12 new Senior ML Engineer roles in Bengaluru", "Based on your profile: Senior ML Engineer at a fintech, Applied Scientist at an e-commerce firm. Apply before Friday.", "Job alert"),
    ("Flash sale: 60% off annual plan, today only", "Upgrade now and save big on our Pro tier. Offer ends at midnight. Use code SAVE60 at checkout.", "Promotion"),
    ("What's new in v3.2: offline mode and faster sync", "We shipped offline mode, 2x faster sync, and a redesigned settings page. Update from the app store.", "Product update"),
]
SUMMARY = [
    "The city council voted on Tuesday to extend the metro's evening service by two hours on weekdays starting next month. Officials said ridership after 9 pm had grown 40 percent in a year, and that the extra trains would be funded by a small rise in parking fees near stations. Some residents near the depots raised concerns about noise.",
    "Researchers trained a small language model only on synthetic textbooks and found it beat models ten times larger on basic coding tests. However, the small model struggled with questions outside the textbook topics, and the authors warn that synthetic data can hide gaps that only show up in real use.",
    "A local bakery started composting its unsold bread and selling the compost to nearby community gardens. In six months it cut its food waste by half and earned enough from compost sales to pay for a part-time delivery helper. The owner now plans to share the process with three other shops.",
    "The monsoon arrived a week later than usual this year, delaying sowing in several districts. Farmers who used short-duration seed varieties were less affected, and the agriculture department says it will widen access to those seeds next season along with weather alerts sent by SMS.",
]
TWEET = [
    ["Tested a 0.6B model vs a frontier model on 40 math problems", "small model 57.5% with 5 votes, frontier 100%", "frontier cost 15 cents for all 40"],
    ["Batched 20 items per API call instead of 1", "cost dropped 8x", "accuracy basically unchanged (89% vs 90.5%)"],
    ["Compared agent context compaction methods", "manual handoff note was 27% cheaper than doing nothing", "no facts lost in any method"],
    ["TF-IDF + logistic regression on news topics", "87.5% accuracy, trained in 1.5 seconds", "within 3 points of a frontier LLM"],
]
REGEX = [
    ("an Indian PIN code: exactly 6 digits, first digit 1-9, nothing else on the line",
     ["500072", "110001", "560037"], ["050072", "50007", "5000721", "50a072"]),
    ("a date in YYYY-MM-DD format with month 01-12 and day 01-31",
     ["2026-10-15", "1999-01-31", "2000-12-01"], ["2026-13-01", "2026-1-05", "2026-10-32", "26-10-15"]),
    ("a hashtag: '#' followed by a letter, then letters, digits or underscores (no spaces)",
     ["#AI", "#tech_in_general", "#GPT4"], ["#4ever", "AI", "# AI", "#"]),
    ("a semantic version like 1.2.3 (three non-negative integers separated by dots, no leading 'v')",
     ["1.2.3", "10.0.11", "0.0.1"], ["1.2", "v1.2.3", "1.2.3.4", "1..3"]),
]
JSONX = [
    ("Acme Analytics is hiring a Senior Data Scientist in Pune. Hybrid, 3 days in office. 6+ years of experience required. Apply by 30 Nov 2026.",
     {"company": "Acme Analytics", "role": "Senior Data Scientist", "city": "Pune", "min_years": 6}),
    ("Role: Machine Learning Engineer | Company: Brightpath Labs | Location: Hyderabad | Experience: 4+ years | Remote-friendly.",
     {"company": "Brightpath Labs", "role": "Machine Learning Engineer", "city": "Hyderabad", "min_years": 4}),
    ("We at Nimbus Health are looking for an AI Research Scientist (Bengaluru) with at least 5 years of industry experience.",
     {"company": "Nimbus Health", "role": "AI Research Scientist", "city": "Bengaluru", "min_years": 5}),
    ("Kestrel Fintech - Applied Scientist - Chennai - minimum 3 years experience - competitive pay.",
     {"company": "Kestrel Fintech", "role": "Applied Scientist", "city": "Chennai", "min_years": 3}),
]

def items():
    out = []
    for i, (s, b, g) in enumerate(TRIAGE):
        out.append(dict(id=f"triage{i}", task="triage", gold=g, prompt=f"Classify this newsletter email into exactly one category: {', '.join(TRIAGE_LABELS)}. Reply with the category only.\n\nSubject: {s}\nPreview: {b}"))
    for i, t in enumerate(SUMMARY):
        out.append(dict(id=f"summary{i}", task="summary", source=t, prompt=f"Summarize this text in one sentence of at most 25 words. Reply with the sentence only.\n\n{t}"))
    for i, pts in enumerate(TWEET):
        out.append(dict(id=f"tweet{i}", task="tweet", source=pts, prompt="Write one tweet (max 280 characters, include exactly one hashtag) for a tech blog, based on these notes. Reply with the tweet only.\n- " + "\n- ".join(pts)))
    for i, (d, pos, neg) in enumerate(REGEX):
        out.append(dict(id=f"regex{i}", task="regex", pos=pos, neg=neg, prompt=f"Write a single Python regular expression that fully matches {d}. Reply with the regex only, no quotes, no code fences, no explanation."))
    for i, (t, g) in enumerate(JSONX):
        out.append(dict(id=f"json{i}", task="json", gold=g, prompt=f"Extract the job posting fields as JSON with keys company (string), role (string), city (string), min_years (integer). Reply with the JSON object only.\n\n{t}"))
    return out
