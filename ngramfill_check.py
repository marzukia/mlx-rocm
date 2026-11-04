import ast

src = open("bench/mtp-bench6.py").read()
tree = ast.parse(src)
cls = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "NgramFill"
)
mod = ast.Module(body=[cls], type_ignores=[])
ns = {}
exec(compile(mod, "NgramFill", "exec"), ns)
NgramFill = ns["NgramFill"]

# degenerate-loop case (the 1919 repeat): key of four 1919s -> follows with 1919
nf = NgramFill([7, 1919, 1919, 1919, 1919, 1919, 1919, 1919], 4)
assert nf.lookup(1, []) == 1919, nf.lookup(1, [])
assert nf.lookup(2, [1919]) == 1919, nf.lookup(2, [1919])
assert nf.lookup(2, [5]) is None
# mixed context
nf2 = NgramFill([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 9], 4)
print("l1:", nf2.lookup(1, []), "l2[2]:", nf2.lookup(2, [2]), "l2[1]:", nf2.lookup(2, [1]))
nf2.update(99)
print("after update l1:", nf2.lookup(1, []))
assert nf2.lookup(1, []) == 99  # tail key (1,2,3,9)? no: last-4 = (2,3,9,?) wait
print("NgramFill OK")
