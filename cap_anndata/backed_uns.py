from typing import List, Any


class CapAnnDataUns(dict):
    __keys_to_remove: List[str] = []

    def __delitem__(self, __key: Any) -> None:
        self.__keys_to_remove.append(__key)
        return super().__delitem__(__key)

    def __setitem__(self, __key: Any, __value: Any) -> None:
        if __key in self.__keys_to_remove:
            self.__keys_to_remove.remove(__key)
        return super().__setitem__(__key, __value)

    @property
    def keys_to_remove(self):
        return self.__keys_to_remove

    def pop(self, __key: Any, __default: Any = None) -> Any:
        if __key in self:
            self.__keys_to_remove.append(__key)
        return super().pop(__key, __default)

    def popitem(self) -> Any:
        item = super().popitem()
        self.__keys_to_remove.append(item[0])
        return item
