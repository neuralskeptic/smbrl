# 1st version: call with parentheses to get typed Decorator
# class Test(Decorator(str)):
#     def capitalize(self):
#         print('decorated!')
#         return self.decorated.capitalize()
# asdf = Test('123')

# def DecoratorFactory(T: type):
#     class Decorator(T):
#         def __init__(self, decorated: T):
#             self.decorated = decorated

#         def __getattr__(self, name):
#             return getattr(self.__dict__['decorated'], name)

#         def __setattr__(self, name, value):
#             if name in ('decorated'):
#                 self.__dict__[name] = value
#             else:
#                 setattr(self.__dict__['decorated'], name, value)

#         def __repr__(self):
#             return (
#                 self.__class__.__name__ +
#                 "<Decorator[" + T.__name__ + "]>" +
#                 "(" + self.decorated.__repr__() + ")"
#             )

#     # if decorated was instantiated, abstract methods must be implemented!
#     Decorator.__abstractmethods__ = frozenset()
#     return Decorator


# 2nd version: call with square brackets to get typed Decorator; wraps `self.decorated`
# class Test(Decorator[str]):
#     def capitalize(self):
#         print('decorated!')
#         return self.decorated.capitalize()
# asdf = Test('123')
from copy import deepcopy


class Decorator:
    # looks and behaves like Generic
    # returns _DecoratorFactory, which catches __new__, creates new class type
    #  inheriting from _Decorator and decorated type (manual mro) and instantiates it
    def __class_getitem__(_, T):
        class _DecoratorFactory:
            def __new__(cls, decorated: T):
                class _Decorator(type(decorated)):  # inherits from decorated type
                    def __init__(self, decorated: T):
                        self.decorated = decorated

                    def __getattr__(self, name):
                        return getattr(self.__dict__["decorated"], name)

                    def __setattr__(self, name, value):
                        if name in ("decorated"):
                            self.__dict__[name] = value
                        else:
                            setattr(self.__dict__["decorated"], name, value)

                    def __repr__(self):
                        # MyDecorator<Decorator[T]>(...)
                        return f"{cls.__name__}<Decorator[{T.__name__}]>({self.decorated.__repr__()})"

                    def __copy__(self):
                        """see: https://stackoverflow.com/a/15774013"""
                        cls = self.__class__
                        result = cls.__new__(cls, self.decorated)  # changed line!!!
                        result.__dict__.update(self.__dict__)
                        return result

                    def __deepcopy__(self, memo):
                        """see: https://stackoverflow.com/a/15774013"""
                        cls = self.__class__
                        result = cls.__new__(cls, self.decorated)  # changed line!!!
                        memo[id(self)] = result
                        for k, v in self.__dict__.items():
                            setattr(result, k, deepcopy(v, memo))
                        return result

                # shorten introspection representation
                _Decorator.__qualname__ = "<...>." + _Decorator.__name__
                # manual instantiation, to have correct mro (inheritance order)
                # 1. inherit from _Decorator (wich inherits from decorated type)
                # 2. copy over new methods and name from concrete decorator type (cls)
                decorator_type = type(cls.__name__, (_Decorator,), dict(cls.__dict__))
                decorator_type.__qualname__ = cls.__qualname__
                # if decorated was instantiated, abstract methods must have been implemented!
                decorator_type.__abstractmethods__ = frozenset()
                return decorator_type(decorated)

        return _DecoratorFactory
