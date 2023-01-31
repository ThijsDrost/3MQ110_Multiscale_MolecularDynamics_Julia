using Meshes
using LinearAlgebra
using Plots
using Statistics
import Base.length

mutable struct Particles
    mass::Vector{Float64}
    location::Vector{Point3}
    last_location::Vector{Point3}
    velocity::Vector{Vec3}
    acceleration::Vector{Vec3}
    number::Int

    function Particles(num::Int, std::Float64)
        masses = ones(Float64, (num,))
        locations = generate_locations(num, 1.2)
        last_location = [Point(0.0, 0.0, 0.0) for _ in 1:num]
        velocities = generate_velocities(masses, std)
        accelerations = [Vec(0.0, 0.0, 0.0) for _ in 1:num]
        Particles(masses, locations, last_location, velocities, accelerations, num)
    end

    function Particles(mass, location, last_location, velocity, acceleration, number)
        if (length(mass) != number)
            throw(ArgumentError("`mass_func` generates Vector of wrong length"))
        end
        if (length(location) != number)
            throw(ArgumentError("`location_func` generates Vector of wrong length"))
        end
        if (length(velocity) != number)
            throw(ArgumentError("`velocity_func` generates Vector of wrong length"))
        end
        if (length(acceleration) != number)
            throw(ArgumentError("`acceleration_func` generates Vector of wrong length"))
        end
        if (length(last_location) != number)
            throw(ArgumentError("`last_location` generates Vector of wrong length"))
        end
        new(mass, location, last_location, velocity, acceleration, number)
    end
end


mutable struct Simulation
    particles::Particles
    dt::Float64
    steps::UInt
    boxsize::Real
    save_every::UInt
    boundary_condition::Union{Nothing, String}
    inv_boxsize::Float64
    
    function Simulation(particles::Particles, dt::Float64, steps::Int, boxsize::Real, save_every::Int, boundary_condition=nothing)
        if boxsize <= 0 
            throw(DomainError("`boxsize` should be bigger than zero"))
        end 
        if particles.number < 1 
            throw(ArgumentError("Number of partilces should be at least one"))
        end 
        if dt <= 0
            throw(DomainError("`dt` should be bigger than zero"))
        end
        new(particles, dt, steps, boxsize, save_every, boundary_condition, 1.0/boxsize)
    end

    function Simulation(;particle_num::Int=2, particle_std::Real=0.1, dt::Real=1e-5, steps::Int=100_000, save_num::Int=10000, boxsize=1, boundary_condition=nothing)
        if save_num > steps
            save_every = 1
        elseif save_num > 0
            save_every = steps ÷ save_num
        else
            throw(DomainError("`save_num` should be bigger than zero"))
        end
        particles = Particles(particle_num, particle_std)
        Simulation(particles, dt, steps, boxsize, save_every, boundary_condition)
    end
end

mutable struct SimulationResults
    potential_energy::Vector{Float64}
    kinetic_energy::Vector{Float64}
    locations::Array{Point3}
    time::Vector{Float64}

    function SimulationResults(number::Int, particles::Int)
        locs = fill(Point3(0,0,0), (number, particles))
        new(zeros(number), zeros(number), locs, zeros(number))
    end
    
    function SimulationResults(sim::Simulation)
        SimulationResults(Int(sim.steps ÷ sim.save_every) + 1, sim.particles.number)        
    end
end

function generate_velocities(masses::Vector{Float64}, std::Float64)
    velocities = [Vec(std*randn(Float64), std*randn(Float64), std*randn(Float64)) for _ in masses]
    centre_of_mass_velocity = sum(masses.*velocities)/sum(masses)
    velocities = [v - centre_of_mass_velocity for v in velocities]
    # velocities = map((v)-> v-centre_of_mass_velocity, velocities)
    return velocities
end

function generate_locations(num::Int, spacing::Real)
    vals = ceil(Int64, num^(1/3))-1
    locations = [Point(0.0, 0.0, 0.0) for _ in 1:num]
    index = 1
    for i = 0:vals
        for j = 0:vals
            for k = 0:vals
                locations[index] = spacing*[i,j,k] .+ 0.5*spacing
                index += 1
                if index > num
                    return locations
                end
            end
        end
    end
end

function kinetic_energy(sim::Simulation)
    # 0.5*m*v^2
    sum(0.5*sim.particles.mass .* map((vec) -> sum(vec.^2), sim.particles.velocity))
end 

function calc_forces(sim::Simulation)
    forces = [Vec3(0.0,0.0,0.0) for _ in 1:sim.particles.number]
    for i in 1:sim.particles.number
        for j in (i+1):sim.particles.number
            rij = sim.particles.location[i] - sim.particles.location[j]
            # TODO: least image convention
            rij2 = sum(rij.^2)
            factor = 4.0*(12 / (rij2^7) - 6.0 / (rij2^4))
            force = factor * rij
            forces[i] += force
            forces[j] -= force
        end
    end
    return forces
end

function calc_potential(sim::Simulation)
    potential = 0.0
    for i in 1:sim.particles.number
        for j in (i+1):sim.particles.number
            rij = sim.particles.location[i] - sim.particles.location[j]
            # TODO: least image convention
            rij2 = sum(rij.^2)
            potential += 4.0*(1.0 / (rij2^6) - 1.0 / (rij2^3))
        end
    end
    return potential
end

function run_simulation!(sim::Simulation, stepper!::Function)
    sim_result = SimulationResults(sim)    
    sim_result.potential_energy[1] = calc_potential(sim)
    sim_result.kinetic_energy[1] = kinetic_energy(sim)
    sim_result.locations[1, :, :] = sim.particles.location
    sim_result.time[1] = 0

    for i in 1:sim.steps
        stepper!(sim)
        if i % sim.save_every == 0
            sim_result.potential_energy[(i÷sim.save_every) + 1] = calc_potential(sim)
            sim_result.kinetic_energy[(i÷sim.save_every) + 1] = kinetic_energy(sim)
            sim_result.locations[(i÷sim.save_every) + 1, :, :] = sim.particles.location
            sim_result.time[(i÷sim.save_every) + 1] = sim.dt*i
        end
        if i % (floor(Int64, sim.steps/100)) == 0
            print("\r", i/floor(Int64, sim.steps/100))
        end
    end
    println("")
    return sim_result
end

function euler_step!(sim::Simulation)
    forces = calc_forces(sim)
    sim.particles.velocity += sim.dt * forces ./ sim.particles.mass
    sim.particles.location += sim.dt * sim.particles.velocity
end


sim = Simulation(particle_num=2, particle_std=0.1, dt=1e-5, steps=10_000_000, save_num=10000)
result = run_simulation!(sim, euler_step!)
println(std(result.potential_energy+result.kinetic_energy))

plot(result.time, result.potential_energy, label="Potential energy")
plot!(result.time, result.kinetic_energy, label="Kinetic energy")
plot!(result.time, result.potential_energy+result.kinetic_energy, label="Total energy")

gui()
readline()
# savefig("plot.png")
