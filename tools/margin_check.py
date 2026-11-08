from src.control.margin import PASS_DEG, balance_margin
from src.io.settings import load


def main() -> None:
    m = balance_margin(load())
    ok = m.pop("pass")
    for name, v in m.items():
        print(f"{name:<16} {v:.4g}")
    print("PASS" if ok else f"FAIL, under {PASS_DEG} deg")


if __name__ == "__main__":
    main()
