# TODO: better organization of knowledge base?
# Knowledge base with all the community information
KNOWLEDGE_BASE = {
    "community_info": {
        "name": "ACME Senior Living",
        "phone": "850-445-8362",
        "address": "145 Fake Street, Charlotte, NC, 28203",
        "smoking_policy": "Outdoor smoking areas",
        "care_types": ["Independent Living", "Assisted Living"],
        "room_types": ["1 Bedroom / 1 Bath", "2 Bedroom / 1.5 Bath", "Studios"],
        "capacity": 60,
        "minimum_age": 60,
        "entrance_fee": 3500,
        "monthly_cost_base": 2000,
        "monthly_cost_assisted": 3000,
        "monthly_cost_independent": 2000,
        "included_in_cost": ["Basic Cable", "Internet/WiFi", "Linen Service", "Breakfast", "Lunch", "Dinner", "Housekeeping"],
        "tour_hours": {
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "start_time": "09:00",
            "end_time": "18:00"
        }
    },
    "amenities": {
        "general": ["Elevators", "Party space", "Exercise pool", "Chef-prepared meals with seasonal ingredients", 
                   "Outdoor seating", "Housekeeping services", "Beauty salon/services", "Gym"],
        "services": ["24-hour staffing", "Bathing assistance", "Errand assistance", "Medication management", 
                    "Shopping assistance", "Dressing assistance", "Eating assistance"],
        "cleaning_services": ["Housekeeping", "Linen services"],
        "activities": ["Arts and crafts", "Book clubs", "Card playing", "Cooking classes", 
                       "Exercise programs", "Game nights", "Movie nights", "Yoga"],
        "dietary_options": ["Diabetic options", "Low sugar/salt", "Vegetarian", "Gluten-free"],
        "room_amenities": ["Air conditioning", "Microwaves", "Private kitchenette", "Walk-in shower", "Furnished Rooms"],
        "religious_services": ["Devotional areas"],
        "dining_areas": ["Dining room", "In-room dining", "Restaurant-style meal service"],
        "outdoor_activities": ["Accompanied walks", "Park visits", "Walking trails", "Day trips"],
        "outdoor_areas": ["Courtyard", "Garden", "Outdoor areas suitable for walking"],
        "fitness_exercise": ["Gym or fitness room", "Exercise pool", "Yoga"],
        "medical_services": {
            "respite_care": True,
            "hospice": False,
            "skilled_nursing": True,
            "adult_day_care": False,
            "physical_therapy": "Onsite physical therapy (third party provider)",
            "speech_therapy": None,  # No information available
            "transportation": ["Scheduled local transportation", "Transportation to medical appointments"],
            "private_aides": True
        }
    },
    "policies": {
        "pets": {
            "allowed": True,
            "types": ["Cats allowed", "Small dogs allowed (under 25 lbs.)", "Service animals allowed", "Fishes", "Small birds"]
        },
        "cars": True,
        "couples": True,
        "wheelchair_accessible": True,
        "vision_impaired_friendly": True,
        "visiting_hours": ["Guests at mealtimes", "Flexible visiting hours", "On-site parking for guests"],
        "security": ["Staff background checks"],
        "lease_term": 12,
        "languages": ["English", "Spanish"],
        "payment_options": {
            "medicaid": True,
            "hud": True,
            "ltc_insurance": False,
            "veterans_benefits": True,
            "bridge_loan": True
        }
    },
    "employment": {
        "careers_page": "https://www.talkfurther.com/events-demo"
    }
}