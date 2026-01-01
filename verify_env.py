import sys
import importlib


def check_package(package_name, specific_version=None):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        status = "OK"
        if specific_version and version != specific_version:
            status = f"MISMATCH (Target: {specific_version})"
        print(f"{package_name:20} | {version:15} | {status}")
        return True
    except ImportError as e:
        print(f"{package_name:20} | {'NOT FOUND':15} | {e}")
        return False
    except Exception as e:
        print(f"{package_name:20} | {'ERROR':15} | {e}")
        return False


print("-" * 60)
print(f"{'Package':20} | {'Version':15} | {'Status'}")
print("-" * 60)

print(f"{'Python':20} | {sys.version.split()[0]:15} | Target: 3.12.x")

check_package("torch", "2.4.1")
check_package("triton", "3.0.0")
check_package("flash_stu")
check_package("jax")
check_package("flax")
check_package("gymnasium")
check_package("numpy")

print("-" * 60)
