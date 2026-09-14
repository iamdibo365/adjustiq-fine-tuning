"""
Synthetic Insurance Support Ticket Generator
=============================================
Generates a JSONL dataset of realistic insurance customer-service tickets
for fine-tuning: {"instruction": <customer message>, "response": <triage + draft reply>}.

No external API needed -- pure templated generation with randomized
entities (names, policy numbers, dollar amounts, dates, vehicle/property
details) so you get thousands of non-duplicate, structurally-consistent
examples in seconds.

Run:
    python generate_synthetic_data.py --n 800 --output ../data/insurance_tickets.jsonl
    python generate_synthetic_data.py --n 800 --output ../data/insurance_tickets.jsonl --seed 7

Optional: bump realism further by post-processing a sample through an LLM
paraphraser (see the --paraphrase_sample flag note at the bottom of this
file) -- not required to get a usable training set.
"""

import argparse
import json
import random
from datetime import datetime, timedelta

FIRST_NAMES = [
    "James", "Maria", "David", "Linda", "Robert", "Patricia", "Michael", "Jennifer",
    "William", "Elizabeth", "Carlos", "Aisha", "Wei", "Fatima", "John", "Sarah",
    "Kevin", "Nancy", "Daniel", "Priya",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis", "Rodriguez",
    "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Lee",
    "Perez", "Thompson", "White", "Harris",
]
CITIES = ["Dallas, TX", "Phoenix, AZ", "Columbus, OH", "Tampa, FL", "Denver, CO",
          "Austin, TX", "Charlotte, NC", "Sacramento, CA", "Seattle, WA", "Atlanta, GA"]
VEHICLES = ["2019 Toyota Camry", "2021 Honda CR-V", "2018 Ford F-150", "2022 Tesla Model 3",
            "2017 Nissan Altima", "2020 Jeep Grand Cherokee", "2016 Subaru Outback"]
CONDITIONS = ["type 2 diabetes management", "a routine physical", "a knee surgery follow-up",
              "a prescription refill", "an ER visit for chest pain", "a specialist referral"]


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_policy_number():
    return f"POL-{random.randint(100000, 999999)}"


def rand_claim_number():
    return f"CLM-{random.randint(1000000, 9999999)}"


def rand_amount(lo=50, hi=15000):
    return round(random.uniform(lo, hi), 2)


def rand_date(days_back_max=120):
    d = datetime.now() - timedelta(days=random.randint(1, days_back_max))
    return d.strftime("%B %d, %Y")


# ---------------------------------------------------------------------------
# Each generator returns (instruction, category, priority, reply_body)
# ---------------------------------------------------------------------------

def gen_claim_status():
    name = rand_name()
    claim = rand_claim_number()
    line = random.choice(["auto", "home", "health"])
    days = random.randint(3, 45)
    instr = (
        f"Hi, this is {name}. I filed a {line} insurance claim ({claim}) {days} days ago "
        f"and haven't heard anything back. Can you tell me the status?"
    )
    reply = (
        f"Hi {name.split()[0]}, thanks for checking in on claim {claim}. I've located it in "
        f"our system and it's currently in adjuster review. I'll escalate for a status update "
        f"and follow up with you within 24 hours."
    )
    priority = "Medium" if days < 20 else "High"
    return instr, "Claims - Status Inquiry", priority, reply


def gen_claim_denial():
    name = rand_name()
    claim = rand_claim_number()
    reason = random.choice(["a pre-existing condition exclusion", "a lapsed policy at time of loss",
                             "insufficient documentation", "a coverage limit already reached"])
    instr = (
        f"My name is {name} and my claim {claim} was just denied. The letter mentioned "
        f"{reason} but didn't really explain it. I don't understand why and I need this resolved."
    )
    reply = (
        f"I'm sorry for the frustration, {name.split()[0]}. Claim {claim} was denied due to "
        f"{reason}. I'm pulling the full adjuster notes now and will send you a plain-language "
        f"breakdown along with your appeal options within 24 hours."
    )
    return instr, "Claims - Denial Explanation", "High", reply


def gen_billing_dispute():
    name = rand_name()
    policy = rand_policy_number()
    amount = rand_amount(20, 600)
    instr = (
        f"This is {name}, policy {policy}. I was charged ${amount:.2f} twice on my last billing "
        f"cycle and I need this fixed and refunded."
    )
    reply = (
        f"Thanks for flagging this, {name.split()[0]}. I've confirmed a duplicate charge of "
        f"${amount:.2f} on policy {policy} and submitted it for refund. You should see the "
        f"correction on your account within 3-5 business days."
    )
    return instr, "Billing - Duplicate Charge", "High", reply


def gen_premium_increase():
    name = rand_name()
    policy = rand_policy_number()
    old = rand_amount(80, 200)
    new = round(old * random.uniform(1.1, 1.4), 2)
    instr = (
        f"Hello, {name} here, policy {policy}. My premium jumped from ${old:.2f} to ${new:.2f} "
        f"this renewal and nobody told me why. Can you explain this increase?"
    )
    reply = (
        f"Hi {name.split()[0]}, I understand the concern about the increase on policy {policy}. "
        f"I'm reviewing the rating factors (claims history, regional risk adjustment, and coverage "
        f"changes) that applied at renewal and will send you an itemized explanation within 2 business days."
    )
    return instr, "Billing - Premium Increase Question", "Medium", reply


def gen_cancellation():
    name = rand_name()
    policy = rand_policy_number()
    reason = random.choice(["switching to a cheaper provider", "selling the insured vehicle",
                             "moving out of state", "no longer needing coverage"])
    instr = (
        f"I'd like to cancel my policy {policy} effective immediately. This is {name}, and "
        f"the reason is {reason}. Please confirm and let me know about any refund."
    )
    reply = (
        f"I've processed the cancellation request for policy {policy}, {name.split()[0]}. "
        f"Since you're canceling mid-term, any unused premium will be refunded on a pro-rated "
        f"basis to your original payment method within 5-7 business days."
    )
    return instr, "Cancellation - Refund Request", "Medium", reply


def gen_coverage_question():
    name = rand_name()
    policy = rand_policy_number()
    scenario = random.choice([
        "a rental car while my car is in the shop",
        "water damage from a burst pipe",
        "my kid borrowing my car and getting into an accident",
        "a tree falling on my garage during a storm",
        "theft of a bike from my garage",
    ])
    instr = (
        f"Quick question -- this is {name}, policy {policy}. Does my policy cover {scenario}? "
        f"I want to know before I file anything."
    )
    reply = (
        f"Good question, {name.split()[0]}. Based on policy {policy}'s current coverage tier, "
        f"I'm confirming the specifics of coverage for {scenario} and will send you the exact "
        f"policy language and any applicable deductible within 1 business day."
    )
    return instr, "Coverage - Pre-Claim Question", "Low", reply


def gen_add_vehicle_driver():
    name = rand_name()
    policy = rand_policy_number()
    vehicle = random.choice(VEHICLES)
    instr = (
        f"Hi, {name} here. I need to add a {vehicle} to my policy {policy} -- I just bought it. "
        f"Can you help me get that added today?"
    )
    reply = (
        f"Congrats on the new vehicle, {name.split()[0]}! I've added the {vehicle} to policy "
        f"{policy} pending your VIN confirmation. Your updated declarations page and any premium "
        f"change will be emailed to you within the hour."
    )
    return instr, "Policy Change - Add Vehicle/Driver", "Medium", reply


def gen_address_change():
    name = rand_name()
    policy = rand_policy_number()
    city = random.choice(CITIES)
    instr = (
        f"This is {name}, policy {policy}. I just moved to {city} and need to update my address "
        f"on file so my coverage and rates are accurate."
    )
    reply = (
        f"Thanks for updating us, {name.split()[0]}. I've updated the address on policy {policy} "
        f"to {city}. Note this may adjust your premium based on regional risk factors -- I'll "
        f"confirm the new rate in your updated declarations page."
    )
    return instr, "Policy Change - Address Update", "Low", reply


def gen_agent_escalation():
    name = rand_name()
    policy = rand_policy_number()
    instr = (
        f"I'm extremely frustrated. This is {name}, policy {policy}. I've called three times "
        f"about the same issue and keep getting different answers. I want to speak to a "
        f"supervisor immediately."
    )
    reply = (
        f"I sincerely apologize for the runaround, {name.split()[0]}. I'm escalating policy "
        f"{policy} to a senior specialist right now and requesting a callback within 2 hours "
        f"so this gets resolved in one conversation."
    )
    return instr, "Escalation - Repeat Contact", "High", reply


def gen_fraud_report():
    name = rand_name()
    policy = rand_policy_number()
    instr = (
        f"This is {name}, policy {policy}. I think someone filed a claim using my policy that "
        f"I never made. I'm worried about fraud on my account."
    )
    reply = (
        f"Thank you for reporting this right away, {name.split()[0]}. I've placed a security "
        f"flag on policy {policy} and routed this to our Special Investigations Unit. You'll "
        f"hear from a fraud specialist within 24 hours."
    )
    return instr, "Fraud - Suspected Unauthorized Claim", "High", reply


def gen_health_claim():
    name = rand_name()
    claim = rand_claim_number()
    condition = random.choice(CONDITIONS)
    amount = rand_amount(80, 4000)
    instr = (
        f"Hi, {name} here. I had {condition} and my claim {claim} for ${amount:.2f} still shows "
        f"'processing' after a month. My provider is asking me to pay out of pocket. Can you help?"
    )
    reply = (
        f"I understand the urgency, {name.split()[0]} -- I've flagged claim {claim} for "
        f"expedited review given the provider is requesting payment. I'll have a status update "
        f"or payment confirmation for you within 48 hours."
    )
    return instr, "Claims - Health Processing Delay", "High", reply


GENERATORS = [
    gen_claim_status, gen_claim_denial, gen_billing_dispute, gen_premium_increase,
    gen_cancellation, gen_coverage_question, gen_add_vehicle_driver, gen_address_change,
    gen_agent_escalation, gen_fraud_report, gen_health_claim,
]


def build_example():
    instr, category, priority, reply_body = random.choice(GENERATORS)()
    response = f"Triage: {category} | Priority: {priority} | Draft reply: {reply_body}"
    return {"instruction": instr, "response": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800, help="Number of examples to generate")
    parser.add_argument("--output", default="../data/insurance_tickets.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    seen = set()
    examples = []
    attempts = 0
    while len(examples) < args.n and attempts < args.n * 20:
        attempts += 1
        ex = build_example()
        if ex["instruction"] in seen:
            continue
        seen.add(ex["instruction"])
        examples.append(ex)

    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {args.output}")
    from collections import Counter
    cats = Counter(json.loads(l)["response"].split("|")[0].replace("Triage:", "").strip()
                    for l in open(args.output))
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Optional realism boost (not required):
# Take a random sample of the generated instructions and run them through
# any LLM you have API access to with a prompt like "rewrite this customer
# complaint in a more natural, less templated voice, keep the same facts."
# Splice the paraphrased versions back in before training. Templated data
# alone is enough to validate the QLoRA pipeline and get a real signal.
# ---------------------------------------------------------------------------
