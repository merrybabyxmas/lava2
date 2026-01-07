import inspect
from peft.tuners.lava.layer import LavaLayer

# LavaLayer의 forward 확인
print("=" * 60)
print("PEFT LAVA Forward Code")
print("=" * 60)
print(inspect.getsource(LavaLayer.forward))
