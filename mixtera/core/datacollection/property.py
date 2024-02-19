from dataclasses import dataclass

from mixtera.core.datacollection.property_type import PropertyType


@dataclass
class Property:
    name: str
    type: PropertyType
    buckets: list[str]
