import json
from typing import Optional

class Conversation(object):
    def __init__(self):
        self.dialogs = []
        self.meta = {}
    def append(self, role, message: str, meta:Optional[dict]=None):
        dialog = {
            "role": role,
            "content": message
        }
        if meta:
            dialog.update(meta)
        self.dialogs.append(dialog)
        
    def add_meta(self, meta: dict):
        self.meta.update(meta)
    
    def sanity_check(self):
        # check if there are repeated roles in the self.dialogs
        roles = [dialog['role'] for dialog in self.dialogs]
        # check if there're two consecutive roles
        for i in range(len(roles)-1):
            if roles[i] == roles[i+1]:
                print(f"Two consecutive roles detected at index {i}, roles[i]: {roles[i]}, roles[i+1]: {roles[i+1]}")
                return False
        return True
    def to_dict(self):
        return {
            "conversations": self.dialogs,
            "meta": self.meta
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())