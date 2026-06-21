import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_EPISODES = 10000

EVENT_TYPES = [
    "Traffic Congestion",
    "Flooding",
    "Road Damage",
    "Accident",
    "Power Outage",
    "Water Leakage",
    "Crowd Overflow",
    "Fire Incident",
    "EMS Delay"
]

WEATHER = [
    "Clear",
    "Cloudy",
    "Light Rain",
    "Heavy Rain"
]

AREA_TYPES = [
    "Residential",
    "Commercial",
    "Industrial",
    "Transit Hub",
    "School Zone",
    "Hospital Zone",
    "Market Area"
]

INTERVENTIONS = [
    "Traffic Diversion",
    "Road Closure",
    "Emergency Crew Dispatch",
    "Water Pump Deployment",
    "Public Alert",
    "Additional Personnel",
    "No Action"
]

MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777


def random_location():
    return (
        MUMBAI_LAT + np.random.uniform(-0.12, 0.12),
        MUMBAI_LON + np.random.uniform(-0.12, 0.12)
    )


def determine_event(
        weather,
        construction,
        festival,
        peak_hour):

    probs = {}

    if weather == "Heavy Rain":
        probs["Flooding"] = 0.35
        probs["Traffic Congestion"] = 0.25
        probs["Road Damage"] = 0.15

    if construction:
        probs["Traffic Congestion"] = probs.get(
            "Traffic Congestion", 0) + 0.25

    if festival:
        probs["Crowd Overflow"] = 0.30
        probs["Traffic Congestion"] = probs.get(
            "Traffic Congestion", 0) + 0.20

    if peak_hour:
        probs["Accident"] = 0.10
        probs["Traffic Congestion"] = probs.get(
            "Traffic Congestion", 0) + 0.25
        probs["EMS Delay"] = 0.10

    if not probs:
        return random.choice(EVENT_TYPES)

    return max(probs, key=probs.get)


def determine_causes(weather,
                     construction,
                     festival,
                     event_type):

    causes = []

    if weather == "Heavy Rain":
        causes.append("Heavy Rain")

    if construction:
        causes.append("Road Construction")

    if festival:
        causes.append("Festival Crowd")

    if event_type == "Accident":
        causes.append("High Traffic Volume")

    while len(causes) < 3:
        causes.append(None)

    return causes[:3]


records = []

start_date = datetime(2024, 1, 1)

for i in range(NUM_EPISODES):

    timestamp = start_date + timedelta(
        hours=random.randint(0, 24 * 730))

    weather = random.choices(
        WEATHER,
        weights=[40, 25, 20, 15]
    )[0]

    construction = random.random() < 0.25
    festival = random.random() < 0.10

    peak_hour = timestamp.hour in [
        8, 9, 10,
        17, 18, 19, 20
    ]

    event_type = determine_event(
        weather,
        construction,
        festival,
        peak_hour
    )

    lat, lon = random_location()

    severity = random.choices(
        ["Low", "Medium", "High"],
        weights=[40, 40, 20]
    )[0]

    citizens_affected = random.randint(
        100,
        5000
    )

    avg_delay = random.randint(
        5,
        120
    )

    response_time = random.randint(
        5,
        90
    )

    intervention = random.choice(
        INTERVENTIONS
    )

    resolution_time = random.randint(
        10,
        300
    )

    outcome_score = round(
        random.uniform(0.4, 1.0),
        2
    )

    cause1, cause2, cause3 = determine_causes(
        weather,
        construction,
        festival,
        event_type
    )

    records.append({

        "episode_id": f"EP_{i+1}",

        "timestamp": timestamp,

        "day_of_week": timestamp.strftime("%A"),

        "month": timestamp.month,

        "latitude": lat,
        "longitude": lon,

        "area_type": random.choice(
            AREA_TYPES
        ),

        "weather": weather,

        "rainfall_mm":
            random.randint(50, 150)
            if weather == "Heavy Rain"
            else random.randint(0, 20),

        "temperature":
            random.randint(20, 38),

        "festival_active": festival,

        "construction_active": construction,

        "peak_hour": peak_hour,

        "event_type": event_type,

        "severity": severity,

        "primary_cause": cause1,

        "secondary_cause": cause2,

        "tertiary_cause": cause3,

        "citizens_affected":
            citizens_affected,

        "avg_delay_minutes":
            avg_delay,

        "response_time_minutes":
            response_time,

        "intervention":
            intervention,

        "personnel_deployed":
            random.randint(2, 50),

        "resolution_time":
            resolution_time,

        "outcome_score":
            outcome_score,

        "retrieval_count": 0,

        "importance_score":
            round(random.uniform(0.3, 1.0), 2),

        "confidence_score":
            round(random.uniform(0.4, 1.0), 2)
    })

df = pd.DataFrame(records)

df.to_csv(
    os.path.join(SCRIPT_DIR, "civicmind_episodes.csv"),
    index=False
)

print(df.head())
print("\nGenerated:", len(df), "episodes")
