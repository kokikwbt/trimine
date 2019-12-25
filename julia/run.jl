import Pkg
Pkg.add("NPZ")
Pkg.add("Distributions")
Pkg.add("SpecialFunctions")

include("trimine.jl")
using NPZ
using .TriMine

# load data
X = npzread("tensor.npy")
println(size(X))

# parameter setting
k = 3
u, v, n = size(X)
amax = 0.001
bmax = 10
gmax = 10
n_iter = 10

# define variables
a = 0.0001
b = 10
g = 10
Z = fill(-1, size(X))
O = zeros(u, k)
A = zeros(v, k)
C = zeros(n, k)
Nk = zeros(k)
Nu = zeros(u)
Nku = zeros(k, u)
Nkv = zeros(k, v)
Nkn = zeros(k, n)

# Initialization
params = TriMine.Params(X, k, u, v, n, amax, bmax, gmax)
vars = TriMine.Vars(a, b, g, Z, O, A, C)
counter = TriMine.Counter(Nk, Nu, Nku, Nkv, Nkn)

# TriMine.infer(params, vars, counter)
TriMine.online_learning(params, vars, counter, n_iter)
TriMine.save_model(params, vars, counter)