"""Weaviate collection schema, shared by the corpus and the quarantine collection.

main.py still defines the same properties inline for the batch ingestion path. That copy is
left alone deliberately; this module exists so the quarantine collection is guaranteed to
match the corpus, which is what makes promoting a document a straight object copy rather
than a re-ingestion.
"""

from typing import Any

import weaviate.classes as wvc

from core.log_config import get_logger

logger = get_logger(__name__)

CORPUS_COLLECTION = "GovDocs"
QUARANTINE_COLLECTION = "Quarantine"

# The department tag on a chunk (and on an officer token) is the exact name of its docs/
# subfolder - not an invented short code - so a future ingestion run keyed off that folder
# name lines up with token scoping and the admin dropdown with zero mapping table to drift.
# "ALL" is not a real department: it is the supervisor/cross-department sentinel that skips
# the department filter entirely (see retrieval/search.py).
DEPARTMENTS = frozenset({
    "Agriculture,_Dairy_Development,_Animal_Husbandry_and_Fisheries_Department",
    "Co-operation,_Textiles_and_Marketing_Department",
    "Environment_Department",
    "Finance_Department",
    "Food,_Civil_Supplies_and_Consumer_Protection_Department",
    "General_Administration_Department",
    "Higher_and_Technical_Education_Department",
    "Home_Department",
    "Housing_Department",
    "Industries,_Energy_and_Labour_Department",
    "Information_Technology_Department",
    "Law_and_Judiciary_Department",
    "Marathi_Language_Department",
    "Medical_Education_and_Drugs_Department",
    "Minorities_Development_Department",
    "Other_Backward_Bahujan_Welfare_Department",
    "Parliamentary_Affairs_Department",
    "Persons_with_Disabilities_Welfare_Department",
    "Planning_Department",
    "Public_Health_Department",
    "Public_Works_Department",
    "Revenue_and_Forest_Department",
    "Rural_Development_Department",
    "School_Education_and_Sports_Department",
    "Skill_Development_and_Entrepreneurship_Department",
    "Social_Justice_and_Special_Assistance_Department",
    "Soil_and_Water_Conservation_Department",
    "Tourism_and_Cultural_Affairs_Department",
    "Tribal_Development_Department",
    "Urban_Development_Department",
    "Water_Resources_Department",
    "Water_Supply_and_Sanitation_Department",
    "Women_and_Child_Development_Department",
    "ALL",
})

# The reference deployment (this repo's live GovDocs corpus) is the Higher & Technical
# Education Department portal, so this is the backfill target and the default for new
# tokens/ingestion until an admin picks otherwise.
DEFAULT_DEPARTMENT = "Higher_and_Technical_Education_Department"

_TEXT_PROPERTIES = [
    "translated_text", "child_text", "parent_context", "document_title",
    "doc_number", "issuing_authority", "document_category", "source_filename",
    "supersedes", "references", "department",
]

DEPARTMENT_PROPERTY = "department"


def collection_properties() -> list:
    props = [
        wvc.config.Property(name=name, data_type=wvc.config.DataType.TEXT)
        for name in _TEXT_PROPERTIES
    ]
    props.insert(5, wvc.config.Property(name="year", data_type=wvc.config.DataType.INT))
    return props


def ensure_collection(weaviate_client: Any, name: str) -> None:
    """Create the collection if it is missing. Safe to call on every request."""
    if weaviate_client.collections.exists(name):
        return
    logger.info(f"Creating Weaviate collection '{name}'")
    weaviate_client.collections.create(name=name, properties=collection_properties())


def ensure_department_property(weaviate_client: Any, name: str) -> None:
    """Add the department property to an existing collection if it predates this feature.

    Safe to call on every startup: collections created after this change already carry
    `department` via collection_properties(), so the config read below is a cheap no-op then.
    """
    if not weaviate_client.collections.exists(name):
        return
    collection = weaviate_client.collections.get(name)
    existing = {p.name for p in collection.config.get().properties}
    if DEPARTMENT_PROPERTY in existing:
        return
    logger.info(f"Adding '{DEPARTMENT_PROPERTY}' property to collection '{name}'")
    collection.config.add_property(
        wvc.config.Property(name=DEPARTMENT_PROPERTY, data_type=wvc.config.DataType.TEXT)
    )
