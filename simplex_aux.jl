#*******************************************************************************
# simplex.jl
#
# This script contains the functions required to run the objective center
# finding algorithm script (objective_simplex.jl)
#*******************************************************************************

#===============================================================================
init_config

Initialize configuration for center finding algorithm
The current setup will create nguesses^2 initial simplex locations within
+/- 10 km of the first center guess
The range can be modified by changing the value 10. in the xinit and yinit
definitions below

Input
xguess - x-location of first center guess (km)
yguess - y-location of first center guess (km)
rmwguess - first-guess radius of maximum tangential winds (km)
nguesses - sqrt(# of grid points to initialize simplexes)
           E.g., nguesses = 3 gives 3x3 initial simplex locations (9 total)

Output
*Note that the length of xinit and yinit correspond to the example provided
above where there are nine (3x3) initial simplex locations (nguesses = 3)

xinit - x-location for each initial center guess (km); length(xinit) = 3
yinit - y-location for each initial center guess (km); length(yinit) = 3
radii - radius rings centered on the first-guess RMW (km); length(radii) = 11
        (default is +/- 5 km in 1 km increments; can be changed in simplex_aux.jl)

===============================================================================#

function init_config(xguess::Real,yguess::Real,rmwguess::Real,nguesses::Real)
    xinit = collect(range(xguess - 10., length=nguesses, stop=xguess + 10.))
    yinit = collect(range(yguess - 10., length=nguesses, stop=yguess + 10.))
    radii = collect(rmwguess - 5.:1.:rmwguess + 5.)
    return xinit, yinit, radii
end

#===============================================================================
nanmeanvt_annulus

Compute the mean tangential wind within an annulus of +/- 2 km from a specified
radius
*Note - The performance of this function can be greatly improved by modifying 
        the code if there are no missing values (NaNs) present in the data 
===============================================================================#

function nanmeanvt_annulus(loc::AbstractVector{Ta},r::Real,u_glob::AbstractArray{Tb,2},
                           v_glob::AbstractArray{Tc,2}) where {Ta<:Real,Tb<:Real,Tc<:Real}
    # Define center from input
    xc = loc[1]
    yc = loc[2]
    # Compute the tangential wind
    vt = -u_glob .* sin.(atan.(ones(length(x)) .* (y .- yc)',(x .- xc) .* ones(length(y))')) +
          v_glob .* cos.(atan.(ones(length(x)) .* (y .- yc)',(x .- xc) .* ones(length(y))'))
    # Create an accessible interpolation object via bi-linear interpolation
    vt_itp = extrapolate(interpolate((x .- xc,y .- yc), vt, Gridded(Linear())), NaN)
    phi = collect(0:pi/180.:2*pi - pi/180.)
    # Define the range of radii for the annulus as +/- 2 km in 1 km intervals
    # rrange_ is shifted back one index to compute drsq
    rrange_ = collect(r - 3.:1.:r + 1.)
    rrange = collect(r - 2.:1.:r + 2.)
    drsq = rrange.^2 - rrange_.^2
    # Define arrays
    vt_rings = Array{Float64}(undef,length(rrange),length(phi))
    rrings = Float64[]
    # Compute the radius-weighted vt and store the radius weights
    for j in eachindex(phi)
        for i in eachindex(rrange)
            @inbounds vt_rings[i,j] = 0.5 * drsq[i] * vt_itp(rrange[i] * cos(phi[j]), rrange[i] * sin(phi[j]))
            isnan(vt_rings[i,j]) ? push!(rrings,NaN) : push!(rrings, 0.5 * drsq[i])
        end
    end
    # Compute the area-averaged vt within the annulus
    vt_bar_annulus = nansum(vt_rings)/nansum(rrings)
    # Return the negative of the mean vt within the annulus
    # Throw error if all values within the annulus were NaN
    return -vt_bar_annulus
end

#===============================================================================
rm_outliers

Remove outliers from the simplex centers
===============================================================================#

function rm_outliers(xinit::AbstractVector{Ta},yinit::AbstractVector{Tb},
                     simplex_xcenters::AbstractArray{Tc,2},
                     simplex_ycenters::AbstractArray{Td,2}) where {Ta<:Real,Tb<:Real,Tc<:Real,Td<:Real}
    conv_xcenters = Float64[]
    conv_ycenters = Float64[]
    xbar_centers = mean(simplex_xcenters)
    ybar_centers = mean(simplex_ycenters)
    stdv_centers = sqrt(var(simplex_xcenters) + var(simplex_ycenters))
        for j in eachindex(yinit)
            for i in eachindex(xinit)
                @inbounds dist = sqrt( (simplex_xcenters[i,j] - xbar_centers)^2 +
                                       (simplex_ycenters[i,j] - ybar_centers)^2 )
                if dist < stdv_centers
                    push!(conv_xcenters,simplex_xcenters[i,j])
                    push!(conv_ycenters,simplex_ycenters[i,j])
                end
            end
        end
    prelim_xbar_centers = mean(conv_xcenters)
    prelim_ybar_centers = mean(conv_ycenters)
    prelim_stdv_centers = sqrt(var(conv_xcenters) + var(conv_ycenters))
    return prelim_xbar_centers, prelim_ybar_centers, prelim_stdv_centers
end

#===============================================================================
nanmeanvt_ring

Compute the mean tangential wind at a specified radius
===============================================================================#

# Define a function to determine the center which maximizes vt

function nanmeanvt_ring(x::AbstractVector{Ta},y::AbstractVector{Tb},prelim_xc::Real,
                        prelim_yc::Real,radius::Real,u::AbstractArray{Tc,2},
                        v::AbstractArray{Td,2}) where {Ta<:Real,Tb<:Real,Tc<:Real,Td<:Real}
    # Compute the tangential wind
    vt = -u .* sin.(atan.(ones(length(x)) .* (y .- prelim_yc)',(x .- prelim_xc) .* ones(length(y))')) +
          v .* cos.(atan.(ones(length(x)) .* (y .- prelim_yc)',(x .- prelim_xc) .* ones(length(y))'))
    # Create an accessible interpolation object via bi-linear interpolation
    vt_itp = extrapolate(interpolate((x .- prelim_xc,y .- prelim_yc), vt, Gridded(Linear())), NaN)
    phi = collect(0:pi/180.:2*pi - pi/180.)
    vt_ring = Array{Float64}(undef,length(phi))
    # Loop over all phi and store vt at given radius
    for j in eachindex(phi)
        @inbounds vt_ring[j] = vt_itp(radius * cos(phi[j]), radius * sin(phi[j]))
    end
    return nanmean(vt_ring)
end

#### COPYING FUNCTIONS FROM JULIAMET TO SIMPLIFY JULIA INSTALL

#==============================================================================
read_ncvars

This function is designed to read in specified variables from NetCDF files.
It will remove singleton dimensions when necessary and replace fill values
with NaNs if necessary.

Format for function use:

Define varnames as a string or an array of strings.
Ex: varnames = "DBZ" -- Single var
Ex: varnames = ["x","y","altitude","DBZ"] -- Array of vars

mask_opt - specify whether missval or fillval should be replaced with NaN -
           default is true
dict_opt - specify if output type should be a dictionary - defualt is false
==============================================================================#

function read_ncvars(ncfile::AbstractString,varnames::AbstractArray,
                     mask_opt::Bool=true,dict_opt::Bool=false)

    # Determine if one or more vars needs to be read in
    # One var
    if length(varnames) == 1
        #println("Reading in " * varnames[1] * " ...")
        vardata = ncread(ncfile, varnames[1])
        if ndims(vardata) == 1
            # Determine if values need to be masked
            if mask_opt == true
                # If they do, determine the fill/miss values for the variable
                fillval = ncgetatt(ncfile,varnames[1],"_FillValue")
                missval = ncgetatt(ncfile,varnames[1],"missing_value")
                # Only replace fill/mask values with NaN if fill/mask values
                # exist for the variable
                if typeof(fillval) != Nothing
                    vardata[findall(in(fillval),vardata)] .= NaN
                elseif typeof(fillval) == Nothing && typeof(missval) != Nothing
                    vardata[findall(in(missval),vardata)] .= NaN
                end
                #println("Succesfully read in " * varnames[1] * "!")
                return vardata
            else
                # If no masking, just return vardata
                #println("Succesfully read in " * varnames[1] * "!")
                return vardata
            end
        # If not 1-d, determine if vardata has a single-dimension
        elseif ndims(vardata) > 1
            # Remove single-dimension from multi-dimensional array
            for i in collect(1:ndims(vardata))
                if size(vardata)[i] == 1
                    vardata = dropdims(vardata,dims=i)
                end
            end
            # Determine if fill values need to be masked
            if mask_opt == true
                # If they do, determine the fill/mask values for the variable
                fillval = ncgetatt(ncfile,varnames[1],"_FillValue")
                missval = ncgetatt(ncfile,varnames[1],"missing_value")
                # Only replace fill/miss values with NaN if fill/miss values
                # exist for the variable
                if typeof(fillval) != Nothing
                    vardata[findall(in(fillval),vardata)] .= NaN
                elseif typeof(fillval) == Nothing && typeof(missval) != Nothing
                    vardata[findall(in(missval),vardata)] .= NaN
                end
                #println("Successfully read in " * varnames[1] * "!")
                return vardata
            else
                # If no masking, just return svardata
                #println("Successfully read in " * varnames[1] * "!")
                return vardata
            end
        end
    # More than one var
    elseif length(varnames) > 1
        # Create an ordered dict for the vars
        varsdata = OrderedDict()
        svarsdata = OrderedDict()
        for var in varnames
            #println("Reading in " * var * " ...")
            varsdata[var] = ncread(ncfile,var)
            # Create a second variable to overwrite data that is squeezed
            svarsdata[var] = ncread(ncfile,var)
            # Determine if any var in varsdata has a single-dimension
            if ndims(varsdata[var]) > 1
                # Remove single-dimension from multi-dimensional array
                for i in collect(1:ndims(varsdata[var]))
                    if size(varsdata[var])[i] == 1
                        svarsdata[var] = dropdims(varsdata[var],dims=i)
                    #else
                        #svarsdata[var] = varsdata[var]
                    end
                end
            #else
                #svarsdata[var] = varsdata[var]
            end
            # Determine if values need to be masked
            if mask_opt == true
                # If they do, determine the fill/miss values for the variable
                fillval = ncgetatt(ncfile,var,"_FillValue")
                missval = ncgetatt(ncfile,var,"missing_value")
                # Only replace fill/miss values with NaN if fill/miss values
                # exist for the variable
                if typeof(fillval) != Nothing
                    svarsdata[var][findall(in(fillval),svarsdata[var])] .= NaN
                elseif typeof(fillval) == Nothing && typeof(missval) != Nothing
                    svarsdata[var][findall(in(missval),svarsdata[var])] .= NaN
                end
            else
                # If no masking, just re-store the data
                svarsdata[var] = svarsdata[var]
            end
        end
        # Determine varsdata output type: Values or OrderedDict
        if dict_opt == true
            for var in varnames
                #println("Successfully read in " * var * "!")
            end
            return svarsdata
        else
            for var in varnames
                #println("Successfully read in " * var * "!")
            end
            return collect(values(svarsdata))
        end
    else
       error("Failed to read in var(s), be sure to define them as a string
              or an array of strings!")
    end
end


#==============================================================================
# closest_ind

Search for the index within a 1-D array which most closely corresponds to the
specified value.
==============================================================================#

function closest_ind(arr::AbstractVector{T},val::Real) where T<:Real
    idiff = findmin(abs.(arr.-val))[2]
end


#==============================================================================
filtnan

Generic function to remove NaNs from an array using filter
=============================================================================#

filtnan(x::AbstractArray{T}) where T<:Real = filter(!isnan,x)

#==============================================================================
nansum

Compute the sum of an array excluding NaNs with the option of specifying the
region of a multi-dimensional array
==============================================================================#

# General nansum function over entire array x

function nansum(x::AbstractArray{T}) where T<:Real
    return length(filtnan(x)) == 0 ? NaN : sum(filtnan(x))
end

# Apply nansum over a specific region in array x and option to squeeze the
# output (sout) -- default behavior is the same as Base.sum

function nansum(x::AbstractArray{T},dims::Int,sout::Bool=false) where T<:Real

    # Compute the sum over the desired region
    xsum = mapslices(nansum,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xsum,dims=dims)
    else
        return xsum
    end
end

#==============================================================================
nanmean

Compute the mean of an array excluding NaNs with the option of specifying the
region of a multi-dimensional array
==============================================================================#

# General nanmean function over entire array x

function nanmean(x::AbstractArray{T}) where T<:Real
    return length(filtnan(x)) == 0 ? NaN : mean(filtnan(x))
end

# Apply nanmean over a specific region in array x and option to squeeze the
# output (sout) -- default behavior is the same as Base.mean

function nanmean(x::AbstractArray{T},dims::Int,sout::Bool=false) where T<:Real

    # Compute the mean over the desired region
    xmean = mapslices(nanmean,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xmean,dims=dims)
    else
        return xmean
    end
end

#==============================================================================
nanvar

Compute the variance of an array excluding NaNs with the option of
specifying the region of a multi-dimensional array
** Note: Using the default call to Base.var which uses a bias corrected
         estimator (1/N-1)
==============================================================================#

# General nanvar function over entire array x

function nanvar(x::AbstractArray{T}) where T<:Real
    return length(filtnan(x)) == 0 ? NaN : var(filtnan(x))
end

# Apply nanvar over a specific region in array x and option to squeeze the
# output (sout) -- default behavior is the same as Base.var

function nanvar(x::AbstractArray{T},dims::Int,sout::Bool=false) where T<:Real

    # Compute the var over the desired region
    xvar = mapslices(nanvar,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xvar,dims=dims)
    else
        return xvar
    end
end

#==============================================================================
nanstd

Compute the standard deviation of an array excluding NaNs with the option of
specifying the region of a multi-dimensional array
** Note: Using the default call to Base.std which uses a bias corrected
         estimator (1/N-1)
==============================================================================#

# General nanstd function over entire array x

function nanstd(x::AbstractArray{T}) where T<:Real
    return length(filtnan(x)) == 0 ? NaN : std(filtnan(x))
end

# Apply nanstd over a specific region in array x and option to squeeze the
# output (sout) -- default behavior is the same as Base.std

function nanstd(x::AbstractArray{T},dims::Int,sout::Bool=false) where T<:Real

    # Compute the std over the desired region
    xstd = mapslices(nanstd,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xstd,dims=dims)
    else
        return xstd
    end
end

#==============================================================================
nanmin

Compute the minimum value of an array excluding NaNs with the option of
specifying the region of a multi-dimensional array
==============================================================================#

# General nanmin function over entire array x

function nanmin(x::AbstractArray{T}) where T<:Real
    return minimum(filtnan(x))
end

# Apply nanmin over a specific region in array x

function nanmin(x::AbstractArray{T},dims::Int,sout::Bool=false) where T<:Real

    # Compute the min over the desired region
    xmin = mapslices(nanmin,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xmin,dims=dims)
    else
        return xmin
    end
end

#==============================================================================
nanmax

Compute the maximum value of an array excluding NaNs with the option of
specifying the region of a multi-dimensional array
==============================================================================#

# General nanmax function over entire array x

function nanmax(x::AbstractArray{T}) where T<:Real
    return maximum(filtnan(x))
end

# Apply nanmax over a specific region in array x

function nanmax(x::AbstractArray{T},dims::Int,sout::Bool=false) where T<:Real

    # Compute the min over the desired region
    xmax = mapslices(nanmax,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xmax,dims=dims)
    else
        return xmax
    end
end

#==============================================================================
nanextrema

Compute the extrema of an array excluding NaNs with the option of
specifying the region of a multi-dimensional array
==============================================================================#

# General nanextrema function over entire array x

function nanextrema(x::AbstractArray{T}) where T<:Real
    return extrema(filtnan(x))
end

# Apply nanextrema over a specific region in array x

function nanextrema(x::AbstractArray{T},dims::Int) where T<:Real

    # Compute the extrema over the desired region
    xextrema = mapslices(nanextrema,x,dims=dims)
    # Squeeze singleton dim if sout = true
    if sout
        return dropdims(xextrema,dims=dims)
    else
        return xextrema
    end
end

#==============================================================================
nanfindmax

Find the maximum value in an array excluding NaNs and return both the value and
corresponding index as a tuple. Can handle multi-dimensional arrays.
This code was taken from Base and modified to allow the presence of NaNs.
* Currently does not support dimension specification.
==============================================================================#

function _nanfindmax(a::AbstractArray{T}, ::Colon) where T<:Real
    p = pairs(a)
    y = iterate(p)
    if y === nothing
        throw(ArgumentError("collection must be non-empty"))
    end
    (mi, m), s = y
    isnan(m) ? m = -Inf : nothing
    i = mi
    while true
        y = iterate(p, s)
        y === nothing && break
        (i, ai), s = y
        isnan(ai) ? ai = -Inf : nothing
        if isless(m, ai)
            m = ai
            mi = i
        end
    end
    return (m, mi)
end

nanfindmax(a) = _nanfindmax(a,:)

#==============================================================================
nanargmax

Find the index corresponding to the maximum value in an array, excluding NaNs.
Can handle multi-dimensional arrays.
This code was taken from Base and modified to allow the presence of NaNs.
* Currently does not support dimension specification.
==============================================================================#

nanargmax(a) = nanfindmax(a)[2]

#==============================================================================
nanfindmin

Find the minimum value in an array excluding NaNs and return both the value and
corresponding index as a tuple. Can handle multi-dimensional arrays.
This code was taken from Base and modified to allow the presence of NaNs.
* Currently does not support dimension specification.
==============================================================================#

function _nanfindmin(a::AbstractArray{T}, ::Colon) where T<:Real
    p = pairs(a)
    y = iterate(p)
    if y === nothing
        throw(ArgumentError("collection must be non-empty"))
    end
    (mi, m), s = y
    i = mi
    while true
        y = iterate(p, s)
        y === nothing && break
        (i, ai), s = y
        if isless(ai,m)
            m = ai
            mi = i
        end
    end
    return (m, mi)
end

nanfindmin(a) = _nanfindmin(a,:)

#==============================================================================
nanargmin

Find the index corresponding to the minimum value in an array, excluding NaNs.
Can handle multi-dimensional arrays.
This code was taken from Base and modified to allow the presence of NaNs.
* Currently does not support dimension specification.
==============================================================================#

nanargmin(a) = nanfindmin(a)[2]