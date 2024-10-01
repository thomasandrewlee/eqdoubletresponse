# GetInstrumentResponseWithDoublets.jl
#=
This code will take the isc catalogs for events over M6 in the period 1936-1941 (when we have HRV analog
digital time series data) and the period 1988-2024 when we have digital HRV data available from IRIS.

The workflow is to:
1) read in HRV data as time series for a given channel SPE SPN SPZ LPE LPN LPZ for the period 1936-1941
2) check over which events are covered by the analog time series data using a rough surface wave travel 
    time calculation based on distance. surface waves are considered because they have similar frequency
    content to the microseism
3) take the covered analog events and search for doublets (within set distance / depth / mag limits)
4) get the data for modern doublets using a fetchdata pull (see if we can get sac directly) (match direction SPZ & LPZ -> BHZ)
5) remove instrument response from the modern data (just make correction of gain manually to get to m/s)
6) calculate the PSD for the surface wave times for both modern and historical
7) calculate the transfer function for these PSDs
=#

## USER STRING
usr_str = "/Users/thomaslee/"

## PACKAGES
push!(LOAD_PATH, string(usr_str,"Research/Julia/MyModules"))
import LeeFunctions
const lf = LeeFunctions
using Seis
using SeisRequests
using Geodesics
using Plots
using JLD
using FFTW
using StatsBase
using Dates
using Interpolations
using ProgressBars
using Geodesics
using NaNStatistics

## SETTINGS
# data output directory
c_dataout = string(usr_str,"Desktop/EQDoub/")
# ISC data files
c_old_ISC = string(usr_str,"Research/FindEQDoublets/ISC_M6_1936_1941.txt")
c_new_ISC = string(usr_str,"Research/FindEQDoublets/ISC_M6_1988_2024.txt")
# HRV data files
c_HRV_old = string(usr_str,"Downloads/HRV_SAC_ANALOG/LPZ/")
c_HRV_save = string(c_dataout,"HRV_1936_1940_LPZ.jld")
# EQ save files
c_oldEQ_save = string(c_dataout,"HRV_1936_1940_LPZ_oldEQ.jld")
# search parameters
deplim = 50
surfvel = 3.0 # surface wave velocity km/s
windowstart = -Dates.Minute(05)
windowend = Dates.Minute(55) # window for surface waves
datathresh = 0.9 # data coverage required in window
hrv_lat = 42.5060
hrv_lon = 71.5580

## LOAD ANALOG DATA SAC
if isfile(c_HRV_save) # read from jld
    # load 
    tmpvar = load(c_HRV_save)
    oldT = tmpvar["oldT"]
    oldD = tmpvar["oldD"]
    oldsamprate = tmpvar["oldsamprate"]
    tmpvar = []
else # read from sac
    # find folders starting with digitseis
    fldrs = readdir(c_HRV_old)
    # throw out non folders
    fldrs = fldrs[findall(map(x->isdir(string(c_HRV_old,fldrs[x])),1:lastindex(fldrs)))]
    # throw out short names
    fldrs = fldrs[findall(map(x->length(fldrs[x])>9,1:lastindex(fldrs)))]
    # keep only "DigitSeis" folders
    fldrs = fldrs[findall(map(x->fldrs[x][1:9]=="DigitSeis",1:lastindex(fldrs)))]
    # initialize vector of traces
    traces = []
    times = []
    # get stitched sac for each
    for i = 1:lastindex(fldrs)
        print(string("  Reading data from ",c_HRV_old,fldrs[i],"\n"))
        Sold = lf.combsac(string(c_HRV_old,fldrs[i]))
        push!(times,lf.gettime(Sold))
        push!(traces,trace(Sold))
    end
    # initialize new data and time vectors
    samprate = unique(map(x->unique(diff(times[x])),1:lastindex(times)))
    if length(samprate)==1
        samprate = samprate[1][1]
    else
        samprate = mode(samprate)
        samprate = samprate[1]
        print("WARNING!!! sample rates are not the same!! using mode! \n")
    end
    minT = minimum(minimum.(times))
    maxT = maximum(maximum.(times))
    oldT = minT:samprate:maxT
    oldD = fill!(Vector{Float64}(undef,length(oldT)),NaN)
    # interpolate and stitch sacs for each folder
    print("Interpolating traces...\n")
    for i in ProgressBar(1:lastindex(times))
        # get bounds
        idxtmp = findall(times[i][1] .<= oldT .<= times[i][end])
        # prep times for interpolation
        t0 = Dates.value.(times[i].-times[i][1])
        t1 = Dates.value.(oldT[idxtmp].-times[i][1])
        # interpolate
        itptmp = LinearInterpolation(t0,traces[i])
        d1 = itptmp(t1)
        # place values
        oldD[idxtmp] .= d1
    end
    # save the data as jld
    save(c_HRV_save,"oldT",oldT,"oldD",oldD,"oldsamprate",samprate)
end

if isfile(c_oldEQ_save)
    tmpvar = load(c_oldEQ_save)
    oldEQtme = tmpvar["oldEQtme"]
    oldEQlat = tmpvar["oldEQlat"]
    oldEQlon = tmpvar["oldEQlon"]
    oldEQdep = tmpvar["oldEQdep"]
    oldEQmag = tmpvar["oldEQmag"]
    oldEQtrace = tmpvar["oldEQtrace"]
    oldEQspect = tmpvar["oldEQspect"]
    oldEQspectF = tmpvar["oldEQspectF"]
    tmpvar = []
else
    ## LOAD ISC CATALOG FOR ANALOG
    # read in the ISC file
    ln = open(c_old_ISC) do f
        readlines(f)
    end
    oldEQtme = []
    oldEQlat = []
    oldEQlon = []
    oldEQdep = []
    oldEQmag = []
    for il = 20:lastindex(ln) # skip header line
        #print(string(il,"\n"))
        commas = findall(map(x->ln[il][x]==',',1:lastindex(ln[il])))
        # try read with subseconds
        hypotime = tryparse(DateTime,ln[il][commas[3]+1:commas[5]-1],dateformat"y-m-d,H:M:S")
        # if not, try without
        if isnothing(hypotime)
            hypotime = tryparse(DateTime,ln[il][commas[3]+1:commas[5]-4],dateformat"y-m-d,H:M:S")
        end
        push!(oldEQtme,hypotime)
        push!(oldEQlat,parse(Float64,ln[il][commas[5]+1:commas[6]-1]))
        push!(oldEQlon,parse(Float64,ln[il][commas[6]+1:commas[7]-1]))
        push!(oldEQdep,parse(Float64,ln[il][commas[7]+1:commas[8]-1]))
        push!(oldEQmag,parse(Float64,ln[il][98:101]))
    end 
    print(string("Read ",length(oldEQtme)," events from ",c_old_ISC,"...\n"))

    ## THROW OUT DEEP EVENTS
    bidx = findall(oldEQdep.>deplim)
    oldEQdep[bidx] = []
    oldEQmag[bidx] = []
    oldEQlat[bidx] = []
    oldEQlon[bidx] = []
    oldEQtme[bidx] = []
    print(string("Threw out ",length(bidx)," events for being too deep. ",length(oldEQtme)," events remaining...\n"))

    ## FIND ANALOG EVENTS COVERED
    oldEQtrace = []
    oldEQspect = []
    oldEQspectF = []
    # setup geodesic
    Ga, Gf = Geodesics.EARTH_R_MAJOR_WGS84, Geodesics.F_WGS84
    print("Finding events with data and calculating PSDs... \n")
    if !isdir(string(c_dataout,"oldevents/"))
        mkdir(string(c_dataout,"oldevents/"))
    end
    for i in 1:lastindex(oldEQtme)
        # get distance
        dtmp = Geodesics.surface_distance(hrv_lon,hrv_lat,oldEQlon[i],oldEQlat[i],Ga)
        dtmp = dtmp/1000 # convert to km from m
        # estimate travel time
        ttime = dtmp/surfvel # seconds
        ttime = convert(Int,round(ttime*1000)) # milliseconds
        # get estimate surface wave arrival window
        stime = oldEQtme[i] + Dates.Millisecond(ttime) + windowstart
        etime = oldEQtme[i] + Dates.Millisecond(ttime) + windowend
        targidx = findall(stime .<= oldT .<= etime)
        # check for data coverage 
        if sum(.!isnan.(oldD[targidx]))/length(targidx) >= datathresh # data is there
            # subtract non-paramteric trend
            tracetmp = oldD[targidx]
            tracetmp = tracetmp .- movmean(tracetmp,convert(Int,round(length(targidx)/10)))
            # calculate fft
            noNaNtrace = oldD[targidx]
            noNaNtrace[isnan.(noNaNtrace)] .= mean(filter(!isnan,noNaNtrace))
            specttmpD = FFTW.rfft(noNaNtrace)
            specttmpF = FFTW.rfftfreq(length(targidx),1/(Dates.value(oldsamprate)/1000))
            # convert to PSD
            specttmpPSD = 2*(1/((Dates.value(oldsamprate)/1000)*length(targidx))).*(abs.(specttmpD).^2)
            # plot and write out waveform and spectras
            hpw = plot(oldT[targidx],tracetmp,lc=:black,legend=false,ylabel="pixels",
                title=string(Dates.format(oldEQtme[i],"yyyy-mm-ddTHH:MM:SS.sss"),
                    "; M",oldEQmag[i],"; (",oldEQlat[i],", ",oldEQlon[i],", ",oldEQdep[i],
                    "); ",round(dtmp),"km -> ",ttime/1000,"s"))
            hps = plot(1 ./specttmpF[2:end],specttmpPSD[2:end],xaxis=:log,yaxis=:log,lc=:black,
                label="raw",xlabel="Period (s)",ylabel="pixels^2/Hz",minorgrid=true) 
            plot!(hps,1 ./specttmpF[2:end],movmean(specttmpPSD[2:end],50),lc=:red,label="smoothed")
                # smoothing is roughly a 1Hz window
            hpall = plot(hpw,hps,layout=grid(2,1),size=(1000,1000))
            display(hpall)
            # check with user if data is appropro
            print(string("\nUse data for event ",i,"/",length(oldEQtme),"?\n\n"))
            usedata = readline()
            if usedata=="y" | usedata=="Y"
                # save plot
                savefig(hpall,string(c_dataout,"oldevents/",Dates.format(oldEQtme[i],"yyyymmddTHHMMSS"),".pdf"))
                # save PSD and trace
                push!(oldEQspect,specttmpPSD)
                push!(oldEQspectF,specttmpF)
                push!(oldEQtrace,tracetmp)
            else
                # fill with NaN for the new fields
                push!(oldEQtrace,[NaN])
                push!(oldEQspect,[NaN])
                push!(oldEQspectF,[NaN])
            end
        else # if there is no data
            # fill with NaN for the new fields to go back and delete later
            push!(oldEQtrace,[NaN])
            push!(oldEQspect,[NaN])
            push!(oldEQspectF,[NaN])
        end
    end
    # delete the dataless events

    # save
    save(c_oldEQ_save,
        "oldEQtme",oldEQtme,
        "oldEQlat",oldEQlat,
        "oldEQlon",oldEQlon,
        "oldEQdep",oldEQdep,
        "oldEQmag",oldEQmag,
        "oldEQtrace",oldEQtrace,
        "oldEQspect",oldEQspect,
        "oldEQspectF",oldEQspectF,
    )
    # report
    print(string("Found ",length(oldEQtme)," events for historical HRV...\n"))
end

## LOAD MODERN ISC CATALOG

## FIND EVENTS WHICH ARE CLOSE TO ANALOG
for i = 1:lastindex(isc_new)
    ## CHECK CLOSENESS

    ## MAKE COMPARISON
    if match
        ## GRAB DATA FROM IRIS

        ## CORRECT GAIN

        ## COMPUTE PSD

        ## COMPARE PSD WITH ANALOG

        ## SAVE TXFR FUNC
    end
end

