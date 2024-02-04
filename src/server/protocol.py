from pydantic import BaseModel
from typing import List, Optional

class ReadDatasetRequest(BaseModel):
    fids: List[str]
    streaming: Optional[bool] = False