## **EEGSubjectProtocol (Protocol)**

**Attributes**
- `state: EEGSubjectStateTracker`  
- `trials: list[EEGTrialProtocol]`  
- `source_filepath: str`  

**Properties / Functions**
- `stratified_folds(self, num_folds: int) -> list[list[EEGTrialProtocol]]`  
- `__init__(self, filepath: str)`  
- `map_labels(self, rule_filepath: str) -> EEGSubjectProtocol`  
- `subaverage(self, size: int) -> EEGSubjectProtocol`  
- `trim(self, start_index: int, end_index: int) -> EEGSubjectProtocol`  

---

## **EEGTrialProtocol (Protocol)**

**Attributes**
- `data: npt.NDArray`  
- `timestamps: npt.NDArray`  
- `trial_index: int`  
- `raw_label: Label`  

**Properties / Functions**
- `mapped_label(self)`  

---

## **EEGSubjectStateTracker**

**Attributes**
- `state_set: set[EEGSubjectState]`  

**Functions**
- `__init__(self)`  
- `mark_subaveraged(self)`  
- `mark_folded(self)`  
- `is_modified(self)`  
- `is_subaveraged(self)`  
- `is_folded(self)`  

---

## **EEGSubjectState (Enum)**

**Members**
- `UNMODIFIED`  
- `SUBAVERAGED`  
- `FOLDED`  
