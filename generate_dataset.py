"""
Customer Call Pattern Analysis - Dataset Generator
Generates 50,000 realistic call records for a Voicebot System
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────
# Configuration Constants
# ──────────────────────────────────────────────
NUM_RECORDS = 50_000
NUM_CUSTOMERS = 12_000
NUM_AGENTS = 80
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 6, 30)

CITIES_STATES = {
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Delhi": "Delhi", "Noida": "Uttar Pradesh", "Gurugram": "Haryana",
    "Bengaluru": "Karnataka", "Mysuru": "Karnataka", "Hubli": "Karnataka",
    "Hyderabad": "Telangana", "Warangal": "Telangana",
    "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Madurai": "Tamil Nadu",
    "Kolkata": "West Bengal", "Howrah": "West Bengal",
    "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan",
    "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh", "Agra": "Uttar Pradesh",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh",
    "Patna": "Bihar", "Guwahati": "Assam", "Bhubaneswar": "Odisha",
    "Chandigarh": "Punjab", "Amritsar": "Punjab",
    "Kochi": "Kerala", "Thiruvananthapuram": "Kerala",
}
CITIES = list(CITIES_STATES.keys())

CITY_WEIGHTS = np.array([
    0.08, 0.05, 0.02,   # Maharashtra
    0.07, 0.04, 0.04,   # Delhi NCR
    0.07, 0.02, 0.01,   # Karnataka
    0.05, 0.02,          # Telangana
    0.05, 0.03, 0.02,   # Tamil Nadu
    0.05, 0.02,          # West Bengal
    0.04, 0.03, 0.02,   # Gujarat
    0.03, 0.02,          # Rajasthan
    0.03, 0.02, 0.02,   # UP
    0.02, 0.02,          # MP
    0.02, 0.01, 0.01,   # East
    0.02, 0.01,          # Punjab
    0.02, 0.01,          # Kerala
])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

VOICEBOT_FLOWS = [
    "Account_Inquiry", "Bill_Payment", "Complaint_Registration",
    "Loan_Application", "Technical_Support", "Policy_Renewal",
    "Appointment_Booking", "Order_Tracking", "Feedback_Collection",
    "Password_Reset", "Balance_Check", "KYC_Verification",
]

CALL_TYPES    = ["Inbound", "Outbound", "Callback"]
CALL_TYPE_W   = [0.70, 0.20, 0.10]

LANGUAGES     = ["Hindi", "English", "Hinglish", "Tamil", "Telugu",
                 "Kannada", "Marathi", "Bengali", "Gujarati", "Malayalam"]
LANGUAGE_W    = [0.30, 0.20, 0.15, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]

CALL_STATUSES = ["Completed", "Failed", "Dropped", "Transferred", "Voicemail"]
STATUS_W      = [0.55, 0.18, 0.12, 0.10, 0.05]

SIP_CODES = {
    "Completed":    [200],
    "Failed":       [486, 503, 408, 480, 487, 500],
    "Dropped":      [487, 408],
    "Transferred":  [302, 200],
    "Voicemail":    [200, 480],
}

DISPOSITIONS = {
    "Completed":    ["Resolved", "Escalated_to_Agent", "Self_Served", "Follow_Up_Scheduled"],
    "Failed":       ["No_Answer", "Busy", "System_Error", "Network_Issue"],
    "Dropped":      ["Customer_Disconnected", "Timeout", "Network_Drop"],
    "Transferred":  ["Transferred_to_Agent", "Transferred_to_Department"],
    "Voicemail":    ["Voicemail_Left", "Voicemail_Full"],
}

FLOW_WEIGHTS = [0.15, 0.14, 0.12, 0.10, 0.09, 0.09,
                0.08, 0.08, 0.06, 0.05, 0.02, 0.02]


# ──────────────────────────────────────────────
# Hour Distribution (realistic business pattern)
# ──────────────────────────────────────────────
HOUR_WEIGHTS = [
    0.005, 0.002, 0.001, 0.001, 0.002, 0.005,   # 00-05
    0.015, 0.030, 0.060, 0.075, 0.080, 0.075,   # 06-11
    0.065, 0.055, 0.060, 0.075, 0.080, 0.075,   # 12-17
    0.060, 0.050, 0.040, 0.030, 0.020, 0.010,   # 18-23
]
HOUR_WEIGHTS = np.array(HOUR_WEIGHTS)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()


def random_datetime(start, end):
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    dt = start + timedelta(seconds=seconds)
    # Override hour using realistic distribution
    hour = np.random.choice(24, p=HOUR_WEIGHTS)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return dt.replace(hour=hour, minute=minute, second=second)


def call_duration(status):
    if status == "Completed":
        return max(30, int(np.random.lognormal(mean=5.2, sigma=0.7)))
    elif status == "Transferred":
        return max(60, int(np.random.lognormal(mean=5.5, sigma=0.6)))
    elif status == "Failed":
        return random.randint(5, 45)
    elif status == "Dropped":
        return random.randint(10, 120)
    else:  # Voicemail
        return random.randint(20, 90)


def generate_dataset():
    print("Generating 50,000 call records...")
    
    customer_ids  = [f"CUST{str(i).zfill(6)}" for i in range(1, NUM_CUSTOMERS + 1)]
    agent_ids     = [f"AGT{str(i).zfill(4)}" for i in range(1, NUM_AGENTS + 1)]
    
    records = []
    for i in range(1, NUM_RECORDS + 1):
        status     = np.random.choice(CALL_STATUSES, p=STATUS_W)
        city       = np.random.choice(CITIES, p=CITY_WEIGHTS)
        state      = CITIES_STATES[city]
        sip_code   = random.choice(SIP_CODES[status])
        disposition= random.choice(DISPOSITIONS[status])
        call_type  = np.random.choice(CALL_TYPES, p=CALL_TYPE_W)
        language   = np.random.choice(LANGUAGES, p=LANGUAGE_W)
        flow       = np.random.choice(VOICEBOT_FLOWS, p=FLOW_WEIGHTS)
        customer   = random.choice(customer_ids)
        dt         = random_datetime(START_DATE, END_DATE)
        duration   = call_duration(status)

        # Outbound calls have a dedicated agent; inbound may or may not
        if call_type == "Outbound":
            agent = random.choice(agent_ids)
        elif status in ("Completed", "Transferred"):
            agent = random.choice(agent_ids) if random.random() < 0.60 else None
        else:
            agent = None

        records.append({
            "call_id":          f"CALL{str(i).zfill(8)}",
            "customer_id":      customer,
            "call_date":        dt.strftime("%Y-%m-%d"),
            "call_time":        dt.strftime("%H:%M:%S"),
            "call_duration":    duration,
            "call_status":      status,
            "sip_response_code":sip_code,
            "city":             city,
            "state":            state,
            "agent_id":         agent if agent else "NO_AGENT",
            "voicebot_flow":    flow,
            "call_type":        call_type,
            "language":         language,
            "call_disposition": disposition,
        })

        if i % 10_000 == 0:
            print(f"  Generated {i:,} records...")

    df = pd.DataFrame(records)

    # ── Introduce 2% realistic missing values ──
    for col in ["agent_id", "call_duration", "language", "voicebot_flow"]:
        mask = np.random.random(len(df)) < 0.015
        df.loc[mask, col] = np.nan

    # ── Introduce ~0.5% duplicates ──
    dup_rows = df.sample(n=250, random_state=99)
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    out_path = "data/call_records.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Dataset saved → {out_path}")
    print(f"   Shape : {df.shape}")
    print(f"\n{df.dtypes}\n")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    df = generate_dataset()
