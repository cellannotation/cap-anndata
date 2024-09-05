from typing import Set, Any


class CapAnnDataDict(dict):
    __keys_to_add: Set[str] = None
    __keys_to_remove: Set[str] = None

    def __delitem__(self, __key: Any) -> None:
        self.keys_to_remove.add(__key)
        return super().__delitem__(__key)

    def __setitem__(self, __key: Any, __value: Any) -> None:
        if __value is not None:
            if __key in self.keys_to_remove:
                self.keys_to_remove.remove(__key)
            self.keys_to_add.add(__key)
        else:
            self.keys_to_remove.add(__key)
            if __key in self.keys_to_add:
                self.keys_to_add.remove(__key) 
        return super().__setitem__(__key, __value)
    
    @property
    def keys_to_add(self) -> Set[str]:
        if self.__keys_to_add is None:
            self.__keys_to_add = set()
        return self.__keys_to_add

    @property
    def keys_to_remove(self) -> Set[str]:
        if self.__keys_to_remove is None:
            self.__keys_to_remove = set()
        return self.__keys_to_remove

    def pop(self, __key: Any, __default: Any = None) -> Any:
        if __key in self:
            self.keys_to_remove.add(__key)
            if __key in self.keys_to_add:
                self.keys_to_add.remove(__key)
        return super().pop(__key, __default)

    def popitem(self) -> Any:
        item = super().popitem()
        key = item[0]
        if key in self.keys_to_add:
            self.keys_to_add.remove(key)
        self.keys_to_remove.add(key)
        return item
