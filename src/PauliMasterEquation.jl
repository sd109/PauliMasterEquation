"""
This file contains the functions to solve for the time dynamics and steady state of a transport model
using a Pauli Master Equation of the form:

    dPn/dt = ∑_m [ W(n, m)P(m, t) - W(m, n)P(n, t) ]

with

    W(n, m) = ∑_μ γ_μ(ω) <m|A_μ|n> <n|A_μ|m>

and ω as the eigen-energy difference ω = εm - εn.
"""


module PauliMasterEquation

export pauli_generator, pauli_steady_state # , pauli_dynamics

using LinearAlgebra, QuantumOpticsBase  # , OrdinaryDiffEq #For solving ODEs

data(O::Operator) = O.data


function W_matrix(H::AbstractOperator, a_ops) #a_ops[i][1] is interaction op and a_ops[i][2] is spectral density

	#Convert to a dense Hermitian array (Herm gives type stable eigen)
	H = Hermitian(data(dense(H)))
    #Get eigenenergy differences
    evals, transf_mat = eigen(H)
    # println("Size(transf_mat) = ", size(transf_mat))
    inv_transf_mat = inv(transf_mat)
    diffs = evals' .- evals #Matrix of eigenenergy differences

    N = length(evals)
    K = length(a_ops)

    #Pre-allocate output matrix
    W_matrix = zeros(N, N)
    for i in 1:K #Loop through a_ops    
        A_eb = inv_transf_mat * data(a_ops[i][1]) * transf_mat
		# W_matrix .+= a_ops[i][2].(diffs) .* A_eb .* transpose(A_eb)
		W_matrix .+= a_ops[i][2].(diffs) .* A_eb .* transpose(conj(A_eb))
	end

    return W_matrix, transf_mat

end


function pauli_generator(H, a_ops)

    L, transf_mat = W_matrix(H, a_ops)
    N = size(L, 1)

    #Add additional required term to each diagonal element of L
    for i in 1:N
        L[i, i] -= sum(L[:, i])
    end

    return L, transf_mat
end


#Versions to convert Ket to dm
pauli_steady_state(H, a_ops, ψ0::Ket; kwargs...) = pauli_steady_state(H, a_ops, dm(ψ0); kwargs...)
pauli_steady_state(H, a_ops, ψ0::Ket, L, transf_mat; kwargs...) = pauli_steady_state(H, a_ops, dm(ψ0), L, transf_mat; kwargs...)



function my_nullspace(
        A::AbstractMatrix; atol::Real = 0.0, rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(atol), 
        alg=LinearAlgebra.QRIteration()
        # alg=LinearAlgebra.DivideAndConquer()
    )

    m, n = size(A)
    (m == 0 || n == 0) && return Matrix{eltype(A)}(I, n, n)
    SVD = svd(A, full=true, alg=alg)
    tol = max(atol, SVD.S[1]*rtol)
    # indstart = sum(s -> s .> tol, SVD.S) + 1
    indstart = findfirst(<(tol), SVD.S) 
    return copy(SVD.Vt[indstart:end,:]')
end


#Version where pauli_generator has been pre-calculated elsewhere
function pauli_steady_state(H, a_ops, ρ0::Operator, L, transf_mat; tol=1e-12)

    N = size(H, 1) #Hilbert space dim
    H_eigenprojectors = (v * v' for v in eachcol(transf_mat)) #Use generator expression to minimize memory burden
    N < 500 && (H_eigenprojectors = collect(H_eigenprojectors)) #Use in-memory array instead of generator for small systems
    Heb_init_pops = map(v -> dot(v', ρ0.data, v), eachcol(transf_mat)) #diag(inv(transf_mat)*ρ0.data*transf_mat)
    
    ρss = zeros(eltype(H), N, N)
    # NS = nullspace(L)
    NS = my_nullspace(L)

    # for (i, v) in enumerate(eachcol(NS))
    for v in eachcol(NS)
        Ps = v * v' * Heb_init_pops
        ρss += mapreduce(*, +, Ps, H_eigenprojectors) # mapreduce works for Arrays and generators
        # ρss += sum(Ps .* H_eigenprojectors)
    end

    if tr(ρss) == 0
        @warn "Nullspace method failed (true steady state is probably orthogonal to initial state) \n -> falling back on slower exponential method."
    else
        return Operator(basis(H), ρss / tr(ρss))
    end


    ### (ROBUST) Using zero-valued eigenvalues of exp(L) method

	#Get inverse eigenbasis transf
	inv_transf_mat = inv(transf_mat)

	#Get initial density matrix in eigenbasis
    ρ0_Heb = inv_transf_mat * data(ρ0) * transf_mat
    init_pops_Heb = real(diag(ρ0_Heb))

    #Construct expL
    expL = exp(L)
    #Diagonalize
	# T = ishermitian(expL) ? Float64 : ComplexF64 # Eigen returns Float if array is herm
    L_transf = eigvecs(expL) #::Eigen{T, T, Array{T, 2}, Array{T, 1}} # Not necessarily Herm so need to enforce type stability
	inv_L_transf = inv(L_transf)
    expL_eb = inv_L_transf * expL * L_transf
    #Find steady state expL in expL's eigenbasis
    expL_eb_ss = convert.(Float64, real(expL_eb) .>= (1.0 - tol)) #This line converts any matrix element that is close to 1 (i.e. exp(0)) to exactly 1
    #Un-diagonalize - the Hamiltonian eigenbasis
    expL_ss = L_transf * expL_eb_ss * inv_L_transf
    #Act this on initial state
    pops_ss_Heb = real(expL_ss * init_pops_Heb)

	#Get steady state density matrix in site basis using population weighted sum of eigenstates
	N = size(transf_mat, 1)
	# eigenstates = Array{ComplexF64, 1}[transf_mat[:, i] for i in 1:N]
	eigenstates = [transf_mat[:, i] for i in 1:N]
	ρ_ss = Operator(H.basis_l, zeros(eltype(H), N, N))
	for i in 1:N
		ρ_ss += Operator(H.basis_l, pops_ss_Heb[i] * eigenstates[i] * conj(transpose(eigenstates[i])))
	end

    return ρ_ss
end


#Version where L needs to be computed
function pauli_steady_state(H, a_ops, ρ0::Operator; kwargs...)

    #Get time evolution generator and eigensystem of H
    L, transf_mat = pauli_generator(H, a_ops)
	return pauli_steady_state(H, a_ops, ρ0, L, transf_mat; kwargs...)

end


# THESE FUNCTIONS BELOW DON'T WORK - PME CAN'T SIMULATE UNITARY DYANMICS (I think?)


#
# function pauli_expLt_solve(H, a_ops, ρ0, t)
#     """
#     NOTE - Need to be careful here with H eigenbasis transformations here because
#     we have to transform using density matrices and then only take the diagonals
#     of the transformed dm as the popuations.
#     Previously, I was trying to transform population vectors as if they were just
#     states which is incorrect.
#     """
#
# 	#Convert ρ0 from QO.jl Ket to simple matrix (conversion of H and a_ops is done in W_matrix function)
# 	ρ0 = dm(ρ0).data
#
#     #Get time evolution generator and eigensystem of H
#     L, transf_mat = pauli_generator(H, a_ops)
#     # evals, transf_mat = eigen(H)
#     inv_transf_mat = inv(transf_mat)
#
#     #Get initial density matrix in eigenbasis
#     ρ0_eb = inv_transf_mat * ρ0 * transf_mat
#     init_pops_eb = diag(ρ0_eb)
#
#     #Calculate populations at time t using generator
#     pops_t_eb = exp(L*t) * init_pops_eb
#     #Convert populations back to density matrix
#     ρt_eb = diagm(pops_t_eb)
#     #Transform back to site basis
#     ρt = transf_mat * ρt_eb * inv_transf_mat
#
# 	# println()
# 	# @show init_pops_eb
# 	# @show pops_t_eb
# 	# @show pops_t
#
#     return real(diag(ρt))
#
# end




# #Function to solve for the site populations vs time
# function pauli_dynamics(tspan, H, a_ops, ρ0::AbstractOperator; kwargs...) #where T <: Number
# 	# ρ0 is initial DENSITY MATRIX, kwargs are passed to ODE solver
# 	ρ0 = dense(ρ0).data
#
# 	# Get RHS L matrix of dPdt = L*P
# 	L, transf_mat = pauli_generator(H, a_ops)
# 	inv_transf_mat = inv(transf_mat)
# 	N, = size(transf_mat) # Hilbert space dims
#
# 	#Transform initial state to Hamiltonian basis and extract populations
# 	ρ0_eb = inv_transf_mat * ρ0 * transf_mat
# 	P0_eb = real(diag(ρ0_eb))
#
# 	#Set up and solve ODE problem
# 	f(P, L, t) = L*P #RHS of ODE
# 	prob = ODEProblem(f, P0_eb, tspan, L)
# 	sol = solve(prob, Tsit5(); kwargs...)
# 	len_times = length(sol.t) #Number of time points in sol
#
# 	#Transform solution back to site basis
# 	# - this unfortunately loses the interpolation info in sol
# 	# - maybe there is a better way round this?
# 	probs = similar(sol.u)
# 	for i in 1:len_times
# 		probs[i] = diag( transf_mat * diagm(sol.u[i]) * inv_transf_mat )
# 	end
#
# 	#Transform back to states for consistency with other master equation types
# 	states = Array{DenseOperator{NLevelBasis{Int}}, 1}(undef, len_times)
# 	for i in 1:L
# 		states[i] = DenseOperator(NLevelBasis(N), diagm(probs[i]))
# 	end
#
# 	return sol.t, states
#
# end



end #module
