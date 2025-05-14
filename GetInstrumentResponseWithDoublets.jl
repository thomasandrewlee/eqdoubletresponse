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
using DSP
using StatsBase
using Dates
using Interpolations
using ProgressBars
using Geodesics
using NaNStatistics
using Measures
using RobustLeastSquares

## SETTINGS
# data output directory
c_dataout = string(usr_str,"Desktop/EQDoub/")
c_runname = "M5.5_LPZ_BHZ_ampscl_stack10_microcorr/" # make sure to add '/' to get folder
# ISC data files
c_old_ISC = string(usr_str,"Research/FindEQDoublets/ISC_M5.5_1935_1941.txt")
c_new_ISC = string(usr_str,"Research/FindEQDoublets/ISC_M5.5_1988_2024.txt")
# HRV data files
c_HRV_old = string(usr_str,"Downloads/HRV_SAC_ANALOG/LPZ/")
c_HRV_save = string(usr_str,"Downloads/1936_40_HRV_TIMESERIES/1936_40_HRV_LPZ_TIMESERIES.jld")
# HRV manual gain correction
station_gains_file = [] # use this empty to avoid correcting gains
station_gains_file = string(usr_str,"Research/HRV_BHZ_Gain.txt")
station_gains_file = string(usr_str,"Research/SACPZ_HRV_19880101_today.txt")
station_gains_SACPZ = true # if the gains are in a SAC PZ format
use_baro = false
# EQ save files
c_oldEQ_save = string(c_dataout,c_runname,"HRV_1936_1940_LPZ_oldEQ.jld")
c_newEQ_save = string(c_dataout,c_runname,"HRV_1988_2023_BHZ_newEQ.jld")
# blacklists (don't use)
ignoreblist = false # go without filter
c_oldEQ_blist = string(c_dataout,"oldblist_LPZ_BHZ.txt")
c_newEQ_blist = string(c_dataout,"newblist_LPZ_BHZ.txt")
c_txfr_blist = string(c_dataout,"txfrblist_LPZ_BHZ.txt")
# microseism climatology
remove_micro_clmt = true
c_micro_clmt = string(usr_str,"Desktop/MicroseismClimatology/Window28_Step3_BB_1_100/HRV.BHZ_1987_2023_Climatology_savefile.jld")
DaysInYear = 365.2422 # tropical year in days
# theoretical transfer function
T0 = 1 # seismometer period (seconds)
Tg = 14 # galvanometer period (seconds)
eg = 20 # galvanometer damping
dampingcritical = false
# search parameters
differentiateold = false # treat old data as displacement to get velocity
usePeriodogram = true # use DSP.periodogram instead of FFTW
stack10 = true # if using periodogram, this is the option to stack 10 minute spectra to match the other processing
weightAmpByMag = true # weight the amplitudes by magnitude (scaled by energy)
weightFreqByAmp = false # weight the contribution of a txfr at a given frequency by the amplitude at that freq
logweights = true # log weights for weighting the frequency median by relative amplitude
deplim = 50 # deepest limit for depth (km)
magmin = 5.5 # smallest allowable mag
distdiff = 50 # allowable distance difference (km)
depdiff = 25 # allowable depth difference (km)
magdiff = 0.3
surfvel = 3.33 # surface wave velocity km/s
oldEQmanualcheck = false # run GUI to check waveforms
oldEQignore = [] # matchID strings to ignore
windowstart = -Dates.Minute(15)
windowend = Dates.Minute(45) # window for surface waves
datathresh = 0.9 # data coverage required in window
hrv_lat = 42.5060
hrv_lon = -71.5580
hrvcha = "BHZ" # channel from HRV to use
smoothing = 0.01 # Hz
# fitting settings
plims = [16 35] # period limits for fitting 
useweights = true # use weights to performs fitting of Q
fullfit = false # do it with everything vs with just the median
# Plotting
plotxlim = (1,100)

## SETUP RUNNAME DIR
if !isdir(string(c_dataout,c_runname))
    mkdir(string(c_dataout,c_runname))
end

## LOAD EVENT BLISTS
if !ignoreblist
    if isfile(c_oldEQ_blist)
        oldEQblist = open(c_oldEQ_blist) do f
            readlines(f)
        end
    end
    if isfile(c_newEQ_blist)
        newEQblist = open(c_newEQ_blist) do f
            readlines(f)
        end
    end
    print(string("Read ",length(oldEQblist)," old and ",length(newEQblist)," new events from blists...\n"))
else
    oldEQblist = ["cat"]
    newEQblist = ["cat"]
end

## OLD EARTHQUAKES
if isfile(c_oldEQ_save)
    tmpvar = load(c_oldEQ_save)
    oldEQtme = tmpvar["oldEQtme"]
    oldEQlat = tmpvar["oldEQlat"]
    oldEQlon = tmpvar["oldEQlon"]
    oldEQdep = tmpvar["oldEQdep"]
    oldEQmag = tmpvar["oldEQmag"]
    oldEQtrace = tmpvar["oldEQtrace"]
    oldEQtraceT = tmpvar["oldEQtraceT"]
    oldEQspect = tmpvar["oldEQspect"]
    oldEQspectF = tmpvar["oldEQspectF"]
    oldEQID = tmpvar["oldEQID"]
    tmpvar = []
    print(string("Loaded ",length(oldEQtme)," old events from ",c_oldEQ_save,"...\n"))
else # build old EQ events
    ## LOAD ANALOG DATA SAC
    if isfile(c_HRV_save) # read from jld
        # load 
        tmpvar = load(c_HRV_save)
        oldT = tmpvar["T"]
        oldD = tmpvar["D"]
        oldsamprate = tmpvar["targetsamplerate"]
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
        oldsamprate = unique(map(x->unique(diff(times[x])),1:lastindex(times)))
        if length(samprate)==1
            oldsamprate = samprate[1][1]
        else
            oldsamprate = mode(samprate)
            oldsamprate = samprate[1]
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
        save(c_HRV_save,"T",oldT,"D",oldD,"targetsamplerate",oldsamprate)
    end
    # differentiate if specified
    if differentiateold
        oldT = oldT[1:end-1] .+ (oldsamprate/2)
        oldD = diff(oldD)
    end

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
    deleteat!(oldEQdep,bidx)
    deleteat!(oldEQmag,bidx)
    deleteat!(oldEQlat,bidx)
    deleteat!(oldEQlon,bidx)
    deleteat!(oldEQtme,bidx)
    print(string("Threw out ",length(bidx)," events for being too deep. ",length(oldEQtme)," events remaining...\n"))

    ## THROW OUT DEEP EVENTS
    bidx = findall(oldEQmag.<magmin)
    deleteat!(oldEQdep,bidx)
    deleteat!(oldEQmag,bidx)
    deleteat!(oldEQlat,bidx)
    deleteat!(oldEQlon,bidx)
    deleteat!(oldEQtme,bidx)
    print(string("Threw out ",length(bidx)," events for being too small. ",length(oldEQtme)," events remaining...\n"))

    ## FIND ANALOG EVENTS COVERED
    oldEQtrace = []
    oldEQtraceT = []
    oldEQspect = []
    oldEQspectF = []
    oldEQID = []
    # setup geodesic
    Ga, Gf = Geodesics.EARTH_R_MAJOR_WGS84, Geodesics.F_WGS84
    print("Finding events with data and calculating PSDs... \n")
    if !isdir(string(c_dataout,c_runname,"oldevents/"))
        mkdir(string(c_dataout,c_runname,"oldevents/"))
    end
    for i in ProgressBar(1:lastindex(oldEQtme))
        # get distance
        dtmp = lf.gcdist(hrv_lon,hrv_lat,oldEQlon[i],oldEQlat[i],Ga)
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
            # get rid of NaN
            noNaNtrace = oldD[targidx]
            noNaNtrace[isnan.(noNaNtrace)] .= mean(filter(!isnan,noNaNtrace))
            # calculate fft / psd
            if usePeriodogram
                if stack10
                    # get integer index window size and step
                    intstep = convert(Int,round(Dates.Minute(1)/oldsamprate))
                    intlen = convert(Int,round(Dates.Minute(10)/oldsamprate))
                    # get 10 minute windows
                    wndwstrt = 1:intstep:lastindex(noNaNtrace)-intlen
                    # initialize periodogram struct
                    dt = Dates.value(oldsamprate)/1000
                    ptmp = DSP.Periodograms.periodogram(rand(intlen+1); fs = 1/dt)
                    specttmpF = DSP.Periodograms.freq(ptmp)
                    psdmatrix = fill!(Array{Float64,2}(undef,(length(specttmpF),length(wndwstrt))),NaN) 
                    # use DSP with 10 minute windows
                    for j = 1:lastindex(wndwstrt)
                        ptmp = DSP.Periodograms.periodogram(noNaNtrace[wndwstrt[j]:wndwstrt[j]+intlen]; fs = 1/dt)
                        psdmatrix[:,j] = DSP.Periodograms.power(ptmp)
                    end
                    # average spectra
                    specttmpPSD = vec(mean(psdmatrix,dims=2))
                else
                    # use DSP
                    dt = Dates.value(oldsamprate)/1000
                    ptmp = DSP.Periodograms.periodogram(noNaNtrace; fs = 1/dt)
                    specttmpF = DSP.Periodograms.freq(ptmp)
                    specttmpPSD = DSP.Periodograms.power(ptmp)
                end
            else
                # use FFTW
                specttmpD = FFTW.rfft(noNaNtrace)
                specttmpF = FFTW.rfftfreq(length(targidx),1/(Dates.value(oldsamprate)/1000))
                # convert to PSD
                specttmpPSD = 2*(1/((Dates.value(oldsamprate)/1000)*length(targidx))).*(abs.(specttmpD).^2)
            end
            # plot and write out waveform and spectras
            hpw = plot(oldT[targidx],tracetmp,lc=:black,legend=false,ylabel="pixels",
                title=string(Dates.format(oldEQtme[i],"yyyy-mm-ddTHH:MM:SS.sss"),
                    "; M",oldEQmag[i],"; (",oldEQlat[i],", ",oldEQlon[i],", ",oldEQdep[i],
                    "); ",round(dtmp),"km (",round(dtmp/111.1,digits=2),"deg) -> ",ttime/1000,"s"))
            hps = plot(1 ./specttmpF[2:end],specttmpPSD[2:end],xaxis=:log,yaxis=:log,lc=:black,
                label="raw",xlabel="Period (s)",ylabel="pixels^2/Hz",minorgrid=true) 
            plot!(hps,1 ./specttmpF[2:end],movmean(specttmpPSD[2:end],50),lc=:red,label="smoothed")
                # smoothing is roughly a 1Hz window
            hpall = plot(hpw,hps,layout=grid(2,1),size=(1000,1000))
            if oldEQmanualcheck
                # check with user if data is appropro
                done = false
                while !done
                    print(string("\nUse data for event ",i,"/",length(oldEQtme),"?\n  "))
                    usedata = readline() # if you error here, you need to run it on the command line
                    if (usedata=="y") | (usedata=="Y")
                        # set done to true
                        done = true
                        # save plot
                        ID = string(Dates.format(oldEQtme[i],"yyyymmdd_HHMM"),"_M",oldEQmag[i])
                        savefig(hpall,string(c_dataout,c_runname,"oldevents/",ID,".pdf"))
                        # save PSD and trace
                        push!(oldEQspect,specttmpPSD)
                        push!(oldEQspectF,specttmpF)
                        push!(oldEQtrace,tracetmp)
                        push!(oldEQtraceT,oldT[targidx])
                        push!(oldEQID,ID)
                    elseif (usedata=="n") | (usedata=="N")
                        # set done to true
                        done = true
                        # fill with NaN for the new fields
                        push!(oldEQtrace,[NaN])
                        push!(oldEQtraceT,[NaN])
                        push!(oldEQspect,[NaN])
                        push!(oldEQspectF,[NaN])
                        push!(oldEQID,"")
                    elseif ((usedata[1]=='f') | (usedata[1]=='F')) | ((usedata[1]=='r') | (usedata[1]=='R'))
                        # move the window
                        tshft = Dates.Minute(parse(Int,usedata[2:end]))
                        if ((usedata[1]=='r') | (usedata[1]=='R')) # back
                            tshft = -tshft
                        end
                        stime = oldEQtme[i] + Dates.Millisecond(ttime) + windowstart + tshft
                        etime = oldEQtme[i] + Dates.Millisecond(ttime) + windowend + tshft
                        targidx = findall(stime .<= oldT .<= etime)
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
                                "); ",round(dtmp),"km (",round(dtmp/111.1,digits=2),"deg) -> ",ttime/1000,"s"))
                        hps = plot(1 ./specttmpF[2:end],specttmpPSD[2:end],xaxis=:log,yaxis=:log,lc=:black,
                            label="raw",xlabel="Period (s)",ylabel="pixels^2/Hz",minorgrid=true) 
                        Nsmth = convert(Int,round(smoothing/mode(diff(specttmpF))))
                        plot!(hps,1 ./specttmpF[2:end],movmean(specttmpPSD[2:end],Nsmth),lc=:red,label="smoothed")
                        hpall = plot(hpw,hps,layout=grid(2,1),size=(1000,1000))
                        display(hpall)
                    else
                        print(string("  !!! unrecognized option: ",usedata,"\n"))
                    end
                end
            else # just add it
                # save plot
                ID = string(Dates.format(oldEQtme[i],"yyyymmdd_HHMM"),"_M",oldEQmag[i])
                savefig(hpall,string(c_dataout,c_runname,"oldevents/",ID,".pdf"))
                # save PSD and trace
                push!(oldEQspect,specttmpPSD)
                push!(oldEQspectF,specttmpF)
                push!(oldEQtrace,tracetmp)
                push!(oldEQtraceT,oldT[targidx])
                push!(oldEQID,ID)
            end
        else # if there is no data
            # fill with NaN for the new fields to go back and delete later
            push!(oldEQtrace,[NaN])
            push!(oldEQtraceT,[NaN])
            push!(oldEQspect,[NaN])
            push!(oldEQspectF,[NaN])
            push!(oldEQID,"")
        end
    end
    # delete the dataless events
    bidx = findall(map(x->isnan(oldEQtrace[x][1])&(length(oldEQtrace[x])==1),1:lastindex(oldEQtrace)))
    deleteat!(oldEQdep,bidx)
    deleteat!(oldEQmag,bidx)
    deleteat!(oldEQlat,bidx)
    deleteat!(oldEQlon,bidx)
    deleteat!(oldEQtme,bidx)
    deleteat!(oldEQtrace,bidx)
    deleteat!(oldEQtraceT,bidx)
    deleteat!(oldEQspect,bidx)
    deleteat!(oldEQspectF,bidx)
    deleteat!(oldEQID,bidx)
    print(string("Removed ",length(bidx)," events without data or nice arrivals, ",
        length(oldEQtme)," events remaining...\n"))
    # save
    save(c_oldEQ_save,
        "oldEQtme",oldEQtme,
        "oldEQlat",oldEQlat,
        "oldEQlon",oldEQlon,
        "oldEQdep",oldEQdep,
        "oldEQmag",oldEQmag,
        "oldEQtrace",oldEQtrace,
        "oldEQtraceT",oldEQtraceT,
        "oldEQspect",oldEQspect,
        "oldEQspectF",oldEQspectF,
        "oldEQID",oldEQID,
    )
    # report
    print(string("Found and saved ",length(oldEQtme)," events for historical HRV...\n"))
end

## NEW EARTHQUAKES
if isfile(c_newEQ_save)
    ## LOAD FROM FILE
    tmpvar = load(c_newEQ_save)
    newEQtme = tmpvar["newEQtme"]
    newEQlat = tmpvar["newEQlat"]
    newEQlon = tmpvar["newEQlon"]
    newEQdep = tmpvar["newEQdep"]
    newEQmag = tmpvar["newEQmag"]
    newEQtrace = tmpvar["newEQtrace"]
    newEQtraceT = tmpvar["newEQtraceT"]
    newEQspect = tmpvar["newEQspect"]
    newEQspectF = tmpvar["newEQspectF"]
    newEQmatch = tmpvar["newEQmatch"]
    newEQID = tmpvar["newEQID"]
    tmpvar = []
else 
    ## LOAD MODERN ISC CATALOG
    # read in the ISC file
    ln = open(c_new_ISC) do f
        readlines(f)
    end
    newEQtme = []
    newEQlat = []
    newEQlon = []
    newEQdep = []
    newEQmag = []
    for il = 30:lastindex(ln) # skip header line
        print(string(il,"\n"))
        if length(ln[il])>97
            commas = findall(map(x->ln[il][x]==',',1:lastindex(ln[il])))
            # try read with subseconds
            hypotime = tryparse(DateTime,ln[il][commas[3]+1:commas[5]-1],dateformat"y-m-d,H:M:S")
            # if not, try without
            if isnothing(hypotime)
                hypotime = tryparse(DateTime,ln[il][commas[3]+1:commas[5]-4],dateformat"y-m-d,H:M:S")
            end
            push!(newEQtme,hypotime)
            push!(newEQlat,parse(Float64,ln[il][commas[5]+1:commas[6]-1]))
            push!(newEQlon,parse(Float64,ln[il][commas[6]+1:commas[7]-1]))
            push!(newEQdep,parse(Float64,ln[il][commas[7]+1:commas[8]-1]))
            push!(newEQmag,parse(Float64,ln[il][98:101]))
        end
    end 
    print(string("Read ",length(newEQtme)," events from ",c_new_ISC,"...\n"))

    ## THROW OUT DEEP EVENTS
    bidx = findall(newEQdep.>deplim)
    deleteat!(newEQdep,bidx)
    deleteat!(newEQmag,bidx)
    deleteat!(newEQlat,bidx)
    deleteat!(newEQlon,bidx)
    deleteat!(newEQtme,bidx)
    print(string("Threw out ",length(bidx)," events for being too deep. ",length(newEQtme)," events remaining...\n"))

    ## THROW OUT DEEP EVENTS
    bidx = findall(newEQmag.<magmin-magdiff)
    deleteat!(newEQdep,bidx)
    deleteat!(newEQmag,bidx)
    deleteat!(newEQlat,bidx)
    deleteat!(newEQlon,bidx)
    deleteat!(newEQtme,bidx)
    print(string("Threw out ",length(bidx)," events for being too small. ",length(newEQtme)," events remaining...\n"))

    ## FIND EVENTS WHICH ARE CLOSE TO ANALOG
    newEQtrace = []
    newEQtraceT = []
    newEQspect = []
    newEQspectF = []
    newEQID = []
    newEQmatch = []
    # check dir for diag
    if !isdir(string(c_dataout,c_runname,"newevents"))
        mkdir(string(c_dataout,c_runname,"newevents"))
    end
    # setup geodesic
    Ga, Gf = Geodesics.EARTH_R_MAJOR_WGS84, Geodesics.F_WGS84
    mindists = []
    for i in ProgressBar(1:lastindex(newEQtme))
        #print(string("i=",i,"\n")) # error on 861
        # check depths
        oldidx = findall(newEQdep[i]-depdiff .<= oldEQdep .<= newEQdep[i]+depdiff)
        # check proximity 
        dtmp = lf.gcdist(newEQlon[i],newEQlat[i],oldEQlon[oldidx],oldEQlat[oldidx],Ga)
        dtmp = dtmp./1000 # m to km
        push!(mindists,minimum(dtmp))
        oldidx2 = oldidx[findall(dtmp.<=distdiff)]
        # check magnitude
        oldidx3 = oldidx2[findall(newEQmag[i]-magdiff .<= oldEQmag[oldidx2] .<= newEQmag[i]+magdiff)]
        if !isempty(oldidx3)
            print(string("i=",i,"\n"))
            # get match info
            matchID = map(x->string(
                    Dates.format(oldEQtme[oldidx3[x]],"yyyymmdd_HHMM"),
                    "_M",oldEQmag[oldidx3[x]]),
                1:lastindex(oldidx3))
            # get distance to HRV
            dHRV = lf.gcdist(hrv_lon,hrv_lat,newEQlon[i],newEQlat[i],Ga)
            dHRV = dHRV/1000 # convert to km from m
            # estimate travel time
            ttime = dHRV/surfvel # seconds
            ttime = convert(Int,round(ttime*1000)) # milliseconds
            # get estimate surface wave arrival window
            stime = newEQtme[i] + Dates.Millisecond(ttime) + windowstart
            etime = newEQtme[i] + Dates.Millisecond(ttime) + windowend
            # grab data from iris
            statmp = get_stations(network="IU", station="HRV", channel=hrvcha,
                starttime=Dates.format(stime,"yyyy-mm-ddTHH:MM:SS"), 
                endtime=Dates.format(etime,"yyyy-mm-ddTHH:MM:SS"))
            if length(statmp)>1 # specify specifically 00 if available
                statmp = get_stations(network="IU", station="HRV", channel="BHZ", location="00",
                    starttime=Dates.format(stime,"yyyy-mm-ddTHH:MM:SS"), 
                    endtime=Dates.format(etime,"yyyy-mm-ddTHH:MM:SS"))
            end
            if !isempty(statmp)
            Stmp = get_data(statmp,stime,etime)
                if !isempty(Stmp)
                    # get trace data and delta out
                    traceTtmp = lf.gettime(Stmp[1])
                    tracetmp = trace(Stmp[1])
                    # initialize new data and time vectors
                    samprate = unique(diff(traceTtmp))
                    if length(samprate)==1
                        samprate = samprate[1]
                    else
                        samprate = mode(samprate)
                        print("WARNING!!! sample rates are not the same!! using mode! \n")
                    end

                    # YOU ARE HERE!!!!

                    # read gain (assuming flat response, not a bad one for BB HRV)
                    if !isempty(station_gains_file)
                        if station_gains_SACPZ # if sac pz files
                            # transform trace into the spectral domain

                            # read gains
                            stimetmp, etimetmp, gaintmp, ptmp, txfrtmp, htmp = lf.readsacpz(
                                        station_gains_file,freqtmp,true,false)
                            # divide by txfrtmp

                            # ifft back to time domain
                            
                        else
                            global gain_stime = [] # start of time periods
                            global gain_etime = [] # end of time periods
                            global gain = [] # gain for that time period (counts / (m/s))
                            if isempty(gain)# if gains file hasn't been read yet
                                # read in the gain file
                                ln = open(station_gains_file) do f
                                    readlines(f)
                                end
                                for il = 3:lastindex(ln) # skip header line
                                    commas = findall(map(x->ln[il][x]==',',1:lastindex(ln[il])))
                                    push!(gain_stime,Dates.DateTime(ln[il][1:commas[1]-1],Dates.dateformat"yyyy-mm-dd"))
                                    push!(gain_etime,Dates.DateTime(ln[il][commas[1]+1:commas[2]-1],Dates.dateformat"yyyy-mm-dd"))
                                    push!(gain,parse(Float64,ln[il][commas[5]+1:commas[6]-1]))
                                end 
                            end
                            # find right gain and correct 
                            traceGtmp = fill!(Vector{Float64}(undef,length(tracetmp)),NaN)
                            for j = 1:lastindex(gain)
                                tidx = findall(gain_stime[j] .<= traceTtmp .<= gain_etime[j])
                                if !isempty(tidx)
                                    traceGtmp[tidx] .= gain[j] 
                                end
                            end
                            tracetmp = tracetmp ./ traceGtmp # convert from counts to m/s
                        end
                    end

                    # correct gain (assuming flat response, not a bad one for VBB HRV)
                    if !isempty(station_gains_file)
                        if station_gains_SACPZ
                            spectG = map(x->fill!(Array{Float64,2}(undef,(length(spectF[x]),length(spectT[x]))),NaN),1:lastindex(spectT))
                        else
                            spectG = map(x->fill!(Array{Float64,2}(undef,length(spectT[x])),NaN),1:lastindex(spectT)) # initialize gain array
                        end
                        for i = 1:lastindex(spectD)
                            if names[i]=="HRV.BHZ" # currently only gains for HRV.BHZ were grabbed
                                if station_gains_SACPZ
                                    # read gains
                                    stimetmp, etimetmp, gaintmp, ptmp, htmp = lf.readsacpz(
                                        station_gains_file,spectF[i],true,true)
                                    if !isdir(string(cDataOut,"responses/"))
                                        mkdir(string(cDataOut,"responses/"))
                                    end
                                    for j = 1:lastindex(htmp)
                                        savefig(htmp[j],string(cDataOut,"responses/",
                                            Dates.format(stimetmp[j],"yyyymmdd"),"_",
                                            Dates.format(etimetmp[j],"yyyymmdd"),".pdf"))
                                    end
                                else
                                    # read in the gain file
                                    ln = open(station_gains_file) do f
                                        readlines(f)
                                    end
                                    stimetmp = [] # start of time periods
                                    etimetmp = [] # end of time periods
                                    gaintmp = [] # gain for that time period (counts / (m/s))
                                    for il = 3:lastindex(ln) # skip header line
                                        commas = findall(map(x->ln[il][x]==',',1:lastindex(ln[il])))
                                        push!(stimetmp,Dates.DateTime(ln[il][1:commas[1]-1],Dates.dateformat"yyyy-mm-dd"))
                                        push!(etimetmp,Dates.DateTime(ln[il][commas[1]+1:commas[2]-1],Dates.dateformat"yyyy-mm-dd"))
                                        push!(gaintmp,parse(Float64,ln[il][commas[5]+1:commas[6]-1]))
                                    end 
                                end
                                # loop over the periods and set gains
                                for j = 1:lastindex(gaintmp)
                                    tidx = findall(stimetmp[j] .<= spectT[i] .<=etimetmp[j])
                                    if !isempty(tidx)
                                        if station_gains_SACPZ
                                            spectG[i][:,tidx] .= gaintmp[j] 
                                        else
                                            spectG[i][tidx] .= gaintmp[j] 
                                        end
                                    end
                                end
                                # divide by gain squared to get (m/s)^2 / Hz from counts^2 / Hz
                                spectD[i] = spectD[i] ./ (spectG[i].^2)
                            else
                                error("STOP! YOU NEED GAINS FOR A STATION THAT IS NOT HRV!!")
                            end
                        end
                    end



                    # subtract non-paramteric trend
                    tracetmp = tracetmp .- movmean(tracetmp,convert(Int,round(length(tracetmp)/10)))
                    # get rid of NaNs
                    noNaNtrace = deepcopy(tracetmp)
                    noNaNtrace[isnan.(noNaNtrace)] .= mean(filter(!isnan,noNaNtrace))
                    # calculate fft
                    if usePeriodogram
                        if stack10
                            # get integer index window size and step
                            intstep = convert(Int,round(Dates.Minute(1)/samprate))
                            intlen = convert(Int,round(Dates.Minute(10)/samprate))
                            # get 10 minute windows
                            wndwstrt = 1:intstep:lastindex(noNaNtrace)-intlen
                            # initialize periodogram struct
                            dt = Dates.value(samprate)/1000
                            ptmp = DSP.Periodograms.periodogram(rand(intlen+1); fs = 1/dt)
                            specttmpF = DSP.Periodograms.freq(ptmp)
                            psdmatrix = fill!(Array{Float64,2}(undef,(length(specttmpF),length(wndwstrt))),NaN) 
                            # use DSP with 10 minute windows
                            for j = 1:lastindex(wndwstrt)
                                ptmp = DSP.Periodograms.periodogram(noNaNtrace[wndwstrt[j]:wndwstrt[j]+intlen]; fs = 1/dt)
                                psdmatrix[:,j] = DSP.Periodograms.power(ptmp)
                            end
                            # average spectra
                            specttmpPSD = vec(mean(psdmatrix,dims=2))
                        else
                            # use DSP
                            dt = Dates.value(samprate)/1000
                            ptmp = DSP.Periodograms.periodogram(noNaNtrace; fs = 1/dt)
                            specttmpF = DSP.Periodograms.freq(ptmp)
                            specttmpPSD = DSP.Periodograms.power(ptmp)
                        end
                    else
                        # use FFTW
                        specttmpD = FFTW.rfft(noNaNtrace)
                        specttmpF = FFTW.rfftfreq(length(noNaNtrace),1/(Dates.value(samprate)/1000))
                        # convert to PSD
                        specttmpPSD = 2*(1/((Dates.value(samprate)/1000)*length(noNaNtrace))).*(abs.(specttmpD).^2)
                    end
                    # plot and write out waveform and spectras
                    hpw = plot(traceTtmp,tracetmp,lc=:black,legend=false,ylabel="(m/s)",
                        title=string(Dates.format(newEQtme[i],"yyyy-mm-ddTHH:MM:SS.sss"),
                            "; M",newEQmag[i],"; (",newEQlat[i],", ",newEQlon[i],", ",newEQdep[i],
                            "); ",round(dHRV),"km (",round(dHRV/111.1,digits=2),"deg) -> ",ttime/1000,"s"))
                    hps = plot(1 ./specttmpF[2:end],specttmpPSD[2:end],xaxis=:log,yaxis=:log,lc=:black,
                        label="raw",xlabel="Period (s)",ylabel="(m/s))^2/Hz",minorgrid=true) 
                    Nsmth = convert(Int,round(smoothing/mode(diff(specttmpF))))
                    plot!(hps,1 ./specttmpF[2:end],movmean(specttmpPSD[2:end],Nsmth),lc=:red,label="smoothed")
                    hpall = plot(hpw,hps,layout=grid(2,1),size=(1000,1000))
                    # save data and plot
                    ID = string(Dates.format(newEQtme[i],"yyyymmdd_HHMM"),"_M",newEQmag[i])
                    savefig(hpall,string(c_dataout,c_runname,"newevents/",ID,".pdf"))
                    push!(newEQtrace,tracetmp)
                    push!(newEQtraceT,traceTtmp)
                    push!(newEQspect,specttmpPSD)
                    push!(newEQspectF,specttmpF)
                    push!(newEQmatch,matchID)
                    push!(newEQID,ID)
                else # no data from IRIS case
                    # fill with NaN for the new fields to go back and delete later
                    push!(newEQtrace,[NaN])
                    push!(newEQtraceT,[NaN])
                    push!(newEQspect,[NaN])
                    push!(newEQspectF,[NaN])
                    push!(newEQmatch,[NaN])
                    push!(newEQID,"")
                end
            else # no station data 
                # fill with NaN for the new fields to go back and delete later
                push!(newEQtrace,[NaN])
                push!(newEQtraceT,[NaN])
                push!(newEQspect,[NaN])
                push!(newEQspectF,[NaN])
                push!(newEQmatch,[NaN])
                push!(newEQID,"")
            end
        else
            # fill with NaN for the new fields to go back and delete later
            push!(newEQtrace,[NaN])
            push!(newEQtraceT,[NaN])
            push!(newEQspect,[NaN])
            push!(newEQspectF,[NaN])
            push!(newEQmatch,[NaN])
            push!(newEQID,"")
        end
    end
    # delete the dataless events
    bidx = findall(map(x->isnan(newEQtrace[x][1])&(length(newEQtrace[x])==1),1:lastindex(newEQtrace)))
    deleteat!(newEQdep,bidx)
    deleteat!(newEQmag,bidx)
    deleteat!(newEQlat,bidx)
    deleteat!(newEQlon,bidx)
    deleteat!(newEQtme,bidx)
    deleteat!(newEQtrace,bidx)
    deleteat!(newEQtraceT,bidx)
    deleteat!(newEQspect,bidx)
    deleteat!(newEQspectF,bidx)
    deleteat!(newEQmatch,bidx)
    deleteat!(newEQID,bidx)
    print(string("Removed ",length(bidx)," events without data or nice arrivals, ",
        length(newEQtme)," events remaining...\n"))
    # save
    save(c_newEQ_save,
        "newEQtme",newEQtme,
        "newEQlat",newEQlat,
        "newEQlon",newEQlon,
        "newEQdep",newEQdep,
        "newEQmag",newEQmag,
        "newEQtrace",newEQtrace,
        "newEQtraceT",newEQtraceT,
        "newEQspect",newEQspect,
        "newEQspectF",newEQspectF,
        "newEQmatch",newEQmatch,
        "newEQID",newEQID,
    )
    # report
    print(string("Found and saved ",length(newEQtme)," events for modern HRV...\n"))
end
print(string())

## DO THE COMPARISON FOR EXISTING MATCHES EMPIRICALLY
print("Finding matches and comparing...\n")
txfrD = []
txfrDwght = []
txfrF = []
txfrID = []
if !isdir(string(c_dataout,c_runname,"txfrs/"))
    mkdir(string(c_dataout,c_runname,"txfrs/"))
end
# load climatology if removing
if remove_micro_clmt
    tmpvar = load(c_micro_clmt)
    Cspectmed = tmpvar["Cspectmed"]
    Cspectf = tmpvar["spectF"]
    CspectTfrac = tmpvar["Twindowctrs"]
    # clear tmpvar
    tmpvar =  []
end
for i in ProgressBar(1:lastindex(newEQtme))
    # print(string("i=",i,"...\n"))
    # check if EQs are on either blist
    if sum(map(x->newEQID[i]==newEQblist[x],1:lastindex(newEQblist))) == 0 # if new ID isn't on the blist
        oldidx = []
        for j = 1:lastindex(newEQmatch[i]) #check against the oldblist
            if sum(map(x->newEQmatch[i][j]==oldEQblist[x],1:lastindex(oldEQblist))) == 0
                tmpidx = findall(map(x->newEQmatch[i][j]==oldEQID[x],1:lastindex(oldEQID)))
                append!(oldidx,tmpidx)
            end
        end
        if !isempty(oldidx) # if there is a match
            for j = 1:lastindex(oldidx)
                # interpolate onto lowest sampled frequency
                minF = maximum([minimum(oldEQspectF[oldidx[j]]),minimum(newEQspectF[i])])
                maxF = minimum([maximum(oldEQspectF[oldidx[j]]),maximum(newEQspectF[i])])
                oldtxfridx = findall(minF .<= oldEQspectF[oldidx[j]] .<= maxF)
                newtxfridx = findall(minF .<= newEQspectF[i] .<= maxF)
                if length(oldtxfridx) == length(newtxfridx) # same
                    txfrFtmp = oldEQspectF[oldidx[j]][oldtxfridx]
                    oldspect = oldEQspect[oldidx[j]][oldtxfridx]
                    if sum(txfrFtmp .== newEQspectF[i][newtxfridx])==length(txfrFtmp) # match is exact
                        newspect = newEQspect[i][newtxfridx]
                    else # interpolation required
                        itp = LinearInterpolation(newEQspectF[i],newEQspect[i])
                        newspect = itp[txfrFtmp]
                    end
                elseif length(oldtxfridx) < length(newtxfridx) # old has lower sample rate
                    txfrFtmp = oldEQspectF[oldidx[j]][oldtxfridx]
                    oldspect = oldEQspect[oldidx[j]][oldtxfridx]
                    itp = LinearInterpolation(newEQspectF[i],newEQspect[i])
                    newspect = itp[txfrFtmp]
                else # new has lower sample rate than old      
                    txfrFtmp = newEQspectF[i][newtxfridx]
                    newspect = newEQspect[i][newtxfridx]
                    itp = LinearInterpolation(oldEQspectF[oldidx[j]],oldEQspect[oldidx[j]])
                    oldspect = itp[txfrFtmp]
                end
                # if not then divide the benioff spectra by the modern velocity spectra
                txfrDtmp = sqrt.(oldspect./newspect)
                # get weighting functions
                scloldspect = lf.unitnorm(oldspect)
                sclnewspect = lf.unitnorm(newspect)
                txfrDwghttmp = (scloldspect .+ sclnewspect)./2
                if logweights
                    txfrDwghttmp = log10.(txfrDwghttmp)
                end
                # get scaling factor we should use (effectively taking out magnitude dependence)
                if weightAmpByMag
                    newAmpScl = 10^newEQmag[i]
                    oldAmpScl = 10^oldEQmag[oldidx[j]]
                    sclfact = newAmpScl / oldAmpScl
                    txfrDtmp = sclfact.*txfrDtmp
                end
                # remove microseism climatology effect if applicable
                if remove_micro_clmt
                    # save the old txfrDtmp
                    global txfrDtmp0 = deepcopy(txfrDtmp)
                    # get the corresponding dates of the old and new spectras as fraction of a year
                    old_Tfrac = rem(Dates.value(oldEQtme[oldidx[j]])/(1000*60*60*24*DaysInYear),1)
                    new_Tfrac = rem(Dates.value(newEQtme[i])/(1000*60*60*24*DaysInYear),1)
                    old_midx = argmin(abs.(CspectTfrac.-old_Tfrac))
                    new_midx = argmin(abs.(CspectTfrac.-new_Tfrac))
                    # get corresponding microseism spectras convert back from log
                    old_mspect = 10 .^(Cspectmed[:,old_midx])   
                    new_mspect = 10 .^(Cspectmed[:,new_midx])
                    # divide to get the microseism seasonality effect
                    mtxfr = sqrt.(old_mspect ./ new_mspect)
                    # interpolate onto the same frequencies as txfr
                    if txfrFtmp != Cspectf
                        # save old one
                        mtxfr0 = deepcopy(mtxfr)
                        # initialize new interpolated mtxfr
                        mtxfr = fill!(Vector{Float64}(undef,length(txfrDtmp)),NaN)
                        # do 1D interpolation
                        itpidx = findall(Cspectf[1] .<= txfrFtmp .<= Cspectf[end])# indices of txfrFtmp overlapping Cspectf
                        itp = LinearInterpolation(Cspectf,mtxfr0)
                        mtxfr[itpidx] = itp(txfrFtmp[itpidx])
                    end
                    # fit by scaling the area under the curve between 2 and 5 seconds 
                    # NOT USING THIS 20250127 TAL
                    # fidx = findall(0.2 .<= txfrFtmp .<= 0.5)
                    # oldarea = sum(oldspect[fidx])
                    # newarea = sum(newspect[fidx])
                    # arearatio = sqrt(oldarea/newarea)
                    # mtxfr1 = mtxfr.*arearatio
                    # make plot of the spectras and microseismtxfr
                    hpm1 = plot(1 ./Cspectf,old_mspect,xaxis=:log,minorgrid=true,xlabel="Period (s)",
                        label=string("Day ",convert(Int,round(new_Tfrac*DaysInYear))),xlim=plotxlim,)
                    plot!(hpm1,1 ./Cspectf,new_mspect,
                        label=string("Day ",convert(Int,round(old_Tfrac*DaysInYear))))
                    # again in log
                    hpm1l = plot(1 ./Cspectf,old_mspect,axis=:log,minorgrid=true,xlabel="Period (s)",
                        label=string("Day ",convert(Int,round(new_Tfrac*DaysInYear))),xlim=plotxlim,)
                    plot!(hpm1l,1 ./Cspectf,new_mspect,
                        label=string("Day ",convert(Int,round(old_Tfrac*DaysInYear))))
                    # make plot of the txfr and microseism correction together
                    hpm2 = plot(1 ./txfrFtmp[2:end],txfrDtmp0[2:end],xaxis=:log,minorgrid=true,
                        label="txfr",xlabel="Period (s)",lc=:black,legend=:topright,
                        y_guidefontcolor=:black,y_foreground_color_axis=:black,
                        y_foreground_color_text=:black,y_foreground_color_border=:black,
                        xlim=plotxlim,)
                    axm2b = twinx(hpm2)
                    plot!(axm2b,1 ./txfrFtmp[2:end],mtxfr[2:end],label="mtxfr",xaxis=:log,
                        y_guidefontcolor=:purple,y_foreground_color_axis=:purple,lc=:purple,legend=:bottomright,
                        y_foreground_color_text=:purple,y_foreground_color_border=:purple,
                        xlim=plotxlim,) 
                    txfrDtmp = txfrDtmp0 ./ mtxfr
                    plot!(hpm2,1 ./txfrFtmp[2:end],txfrDtmp[2:end],label="mtxfr rmvd",lc=:gray) 
                    # again in log
                    hpm2l = plot(1 ./txfrFtmp[2:end],txfrDtmp0[2:end],axis=:log,minorgrid=true,
                        label="txfr",xlabel="Period (s)",lc=:black,legend=:topright,
                        y_guidefontcolor=:black,y_foreground_color_axis=:black,
                        y_foreground_color_text=:black,y_foreground_color_border=:black,
                        xlim=plotxlim,)
                    axm2bl = twinx(hpm2l)
                    plot!(axm2bl,1 ./txfrFtmp[2:end],mtxfr[2:end],label="mtxfr",axis=:log,
                        y_guidefontcolor=:purple,y_foreground_color_axis=:purple,lc=:purple,legend=:bottomright,
                        y_foreground_color_text=:purple,y_foreground_color_border=:purple,
                        xlim=plotxlim,) 
                    txfrDtmp = txfrDtmp0 ./ mtxfr
                    plot!(hpm2l,1 ./txfrFtmp[2:end],txfrDtmp[2:end],label="mtxfr rmvd",lc=:gray)  
                    # aggregate
                    hpm_all = plot(hpm1,hpm1l,hpm2,hpm2l,layout=grid(2,2),size=(1000,600),bottom_margin=5mm)
                    savefig(hpm_all,string(c_dataout,c_runname,"txfrs/",oldEQID[oldidx[j]],"_",newEQID[i],"_micro.pdf"))
                end
                # make plot of both waveforms, both spectras, and txfr
                hpw = plot(Dates.value.(oldEQtraceT[oldidx[j]].-oldEQtraceT[oldidx[j]][1])/(1000*60),
                    oldEQtrace[oldidx[j]],lc=:blue,
                    label=oldEQID[oldidx[j]],ylabel="pixels",la=0.5,xlabel="Minutes",
                    y_guidefontcolor=:blue,y_foreground_color_axis=:blue,
                    y_foreground_color_text=:blue,y_foreground_color_border=:blue,)
                    # x_guidefontcolor=:white,x_foreground_color_axis=:white,
                    # x_foreground_color_text=:white,x_foreground_color_border=:white)
                ax2w = twinx(hpw)
                plot!(ax2w,Dates.value.(newEQtraceT[i].-newEQtraceT[i][1])/(1000*60),newEQtrace[i],lc=:red,
                    label=newEQID[i],ylabel="m/s",la=0.5,legend=:bottomright,
                    y_guidefontcolor=:red,y_foreground_color_axis=:red,
                    y_foreground_color_text=:red,y_foreground_color_border=:red,)
                hps = plot(1 ./txfrFtmp[2:end],oldspect[2:end],xaxis=:log,yaxis=:log,xminorgrid=true,
                    label=oldEQID[oldidx[j]],lc=:blue,ylabel="pixels^2/Hz",xlabel="Period (s)",la=0.5,
                    y_guidefontcolor=:blue,y_foreground_color_axis=:blue,xlim=plotxlim,
                    y_foreground_color_text=:blue,y_foreground_color_border=:blue)
                ax2s = twinx(hps)
                plot!(ax2s,1 ./txfrFtmp[2:end],newspect[2:end],xaxis=:log,yaxis=:log,
                    label=newEQID[i],lc=:red,ylabel="(m/s)^2/Hz",legend=:bottomright,la=:0.5,
                    y_guidefontcolor=:red,y_foreground_color_axis=:red,xlim=plotxlim,
                    y_foreground_color_text=:red,y_foreground_color_border=:red)
                hpt = plot(1 ./txfrFtmp[2:end],txfrDtmp[2:end],xaxis=:log,yaxis=:log,xlim=plotxlim,
                    xminorgrid=true,lc=:black,ylabel="pixels / (m/s)",xlabel="Period (s)",label="raw")
                Nsmth = convert(Int,round(smoothing/mode(diff(txfrFtmp))))
                plot!(hpt,1 ./txfrFtmp[2:end],movmean(txfrDtmp[2:end],Nsmth),lc=:red,label="smoothed")
                hpall = plot(hpw,hps,hpt,layout=grid(3,1),size=(1000,1000))
                savefig(hpall,string(c_dataout,c_runname,"txfrs/",oldEQID[oldidx[j]],"_",newEQID[i],".pdf"))
                # save txfr
                fidx = findall(.!isnan.(txfrDtmp))
                push!(txfrD, txfrDtmp[fidx])
                push!(txfrDwght, txfrDwghttmp[fidx])
                push!(txfrF, txfrFtmp[fidx])
                push!(txfrID, string(oldEQID[oldidx[j]],"_",newEQID[i]))
            end
        end
    end
end

# get bad matches
if !ignoreblist
    if isfile(c_txfr_blist)
        txfrblist = open(c_txfr_blist) do f
            readlines(f)
        end
    end
    print(string("Read ",length(txfrblist)," matches to ignore from blist...\n"))
else
    txfrblist = ["cat"]
end
## AVERAGE TXFR FUNCS
print("Averaging transfer functions...\n")
# get most common freq vector
TXFRF = mode(txfrF)
TXFRM = fill!(Array{Float64,2}(undef,(length(TXFRF),length(txfrD))),NaN)
TXFRMwght = fill!(Array{Float64,2}(undef,(length(TXFRF),length(txfrD))),NaN)
for i = 1:lastindex(txfrD)
    # check against blist
    if sum(map(x->txfrID[i]==txfrblist[x],1:lastindex(txfrblist))) == 0
        # check if interpolation is needed
        global doInterpolation = true
        if length(TXFRF) == length(txfrF[i]) # same
            if sum(txfrF[i] .== TXFRF)==length(txfrF[i]) # match is exact
                TXFRM[:,i] = txfrD[i] 
                TXFRMwght[:,i] = txfrDwght[i]
                global doInterpolation = false
            end
        end
        if doInterpolation # inteprolation required    
            gidx = findall(TXFRF[1] .<= txfrF[i] .<= TXFRF[end])
            gidx2 = map(x->argmin(abs.(txfrF[i][gidx[x]].-TXFRF)),1:lastindex(gidx))
            itp = LinearInterpolation(txfrF[i],txfrD[i])
            itpwght = LinearInterpolation(txfr[i],txfrDwght[i])
            TXFRM[gidx2,i] = itp(TXFRF[gidx2])
            TXFRMwght[gidx2,i] = itp(TXFRF[gidx2])
        end
    end
end

# get median txfr
if weightFreqByAmp  # perform weighted median and percentiles
    gidx = map(x->findall((.!isnan.(TXFRM[x,:])).&(.!isnan.(TXFRMwght[x,:]))),1:lastindex(TXFRF))
    Nsmth = convert(Int,round(smoothing/mode(diff(TXFRF))))
    TXFRD = map(x->lf.wghtdprctle(TXFRM[x,gidx[x]],TXFRMwght[x,gidx[x]],50,true),1:lastindex(TXFRF))
    TXFRD_smth = movmean(TXFRD,Nsmth)
    TXFRD5 = map(x->lf.wghtdprctle(TXFRM[x,gidx[x]],TXFRMwght[x,gidx[x]],5,true),1:lastindex(TXFRF))
    TXFRD5_smth = movmean(TXFRD5,Nsmth)
    TXFRD95 = map(x->lf.wghtdprctle(TXFRM[x,gidx[x]],TXFRMwght[x,gidx[x]],95,true),1:lastindex(TXFRF))
    TXFRD95_smth = movmean(TXFRD95,Nsmth)
else # perform the standard mean
    Nsmth = convert(Int,round(smoothing/mode(diff(TXFRF))))
    TXFRD = map(x->median(filter(!isnan,TXFRM[x,:])),1:lastindex(TXFRF))
    TXFRD_smth = movmean(TXFRD,Nsmth)
    TXFRD5 = map(x->percentile(filter(!isnan,TXFRM[x,:]),5),1:lastindex(TXFRF))
    TXFRD5_smth = movmean(TXFRD5,Nsmth)
    TXFRD95 = map(x->percentile(filter(!isnan,TXFRM[x,:]),95),1:lastindex(TXFRF))
    TXFRD95_smth = movmean(TXFRD95,Nsmth)
end
# plot transfers
hpt = plot(1 ./TXFRF[2:end],TXFRM[2:end,:],label="",title=c_runname,
    xaxis=:log,yaxis=:log,xminorgrid=true,la=0.02,xlim=plotxlim,
    xlabel="Period (s)",ylabel="pixels / (m/s)")
plot!(hpt,1 ./TXFRF[2:end],TXFRD5_smth[2:end],lc=:gray54,ls=:dash,label="",)  
plot!(hpt,1 ./TXFRF[2:end],TXFRD95_smth[2:end],lc=:gray54,ls=:dash,label="",)
plot!(hpt,1 ./TXFRF[2:end],TXFRD[2:end],lc=:black,label="",) 
plot!(hpt,1 ./TXFRF[2:end],TXFRD_smth[2:end],lc=:darkred,label="",)
savefig(hpt,string(c_dataout,c_runname,"txfr.pdf"))
# plot as heatmap
grdamp = range(log10(minimum(filter(!isnan,TXFRM))),
    log10(maximum(filter(!isnan,TXFRM))),51)
grdp = range(maximum([minimum(1 ./TXFRF),plotxlim[1]]),
    minimum([maximum(1 ./TXFRF[TXFRF.>0]),plotxlim[2]]),51)
TXFRGRD = zeros(length(grdamp)-1,length(grdp)-1)
for i = 2:lastindex(grdp)
    fidx = findall(1/grdp[i] .<= TXFRF .<= 1/grdp[i-1])
    if !isempty(fidx)
        TXFRGRD[:,i-1] = map(x->sum(grdamp[x-1] .<= log10.(TXFRM[fidx,:]) .<= grdamp[x]),2:lastindex(grdamp))
    end
end
# normalize by period (since the bands are uneven)
TXFRGRD = TXFRGRD ./ (sum(TXFRGRD,dims=1).+1) # avoid dividing by 0
hptg = heatmap(grdp,grdamp,TXFRGRD,xlabel="Period (s)",ylabel="Log Amplitude",xlim=plotxlim,)
plot!(hptg,1 ./TXFRF[2:end],log10.(TXFRD_smth[2:end]),lc=:lightgray,label="Empirical",lw=2.5)
savefig(hptg,string(c_dataout,c_runname,"txfrgrd.pdf"))
# do the same thing with a log scale
grdamp = range(log10(minimum(filter(!isnan,TXFRM))),
    log10(maximum(filter(!isnan,TXFRM))),51)
grdpl = range(log10(maximum([minimum(1 ./TXFRF),plotxlim[1]])),
    log10(minimum([maximum(1 ./TXFRF[TXFRF.>0]),plotxlim[2]])),51)
TXFRGRDl = zeros(length(grdamp)-1,length(grdpl)-1)
for i = 2:lastindex(grdpl)
    fidx = findall(1/(10^grdpl[i]) .<= TXFRF .<= 1/(10^grdpl[i-1]))
    if !isempty(fidx)
        TXFRGRDl[:,i-1] = map(x->sum(grdamp[x-1] .<= log10.(TXFRM[fidx,:]) .<= grdamp[x]),2:lastindex(grdamp))
    end
end
# normalize by period (since the bands are uneven)
TXFRGRDl = TXFRGRDl ./ (sum(TXFRGRDl,dims=1).+1) # avoid dividing by 0
hptgl = heatmap(grdpl[2:end],grdamp[2:end],TXFRGRDl,xlabel="Log Period (s)",ylabel="Log Amplitude",xlim=log10.(plotxlim),)
plot!(hptgl,log10.(1 ./TXFRF[2:end]),log10.(TXFRD_smth[2:end]),lc=:lightgray,label="Empirical",lw=2.5)
savefig(hptgl,string(c_dataout,c_runname,"txfrgrdl.pdf"))
# plot in a plain way
pltidx = findall(plotxlim[1] .<= (1 ./TXFRF) .<= plotxlim[2])
hpt2l = plot(1 ./TXFRF[pltidx],TXFRD[pltidx,:],label="",title=c_runname,
    xaxis=:log,yaxis=:log,xminorgrid=true,lc=:black,xlim=plotxlim,
    xlabel="Period (s)",ylabel="pixels / (m/s)")    
plot!(hpt2l,1 ./TXFRF[pltidx],TXFRD_smth[pltidx],lc=:darkred,label="",)
hpt2 = plot(1 ./TXFRF[pltidx],TXFRD[pltidx,:],label="",title=c_runname,
    xminorgrid=true,lc=:black,xlim=plotxlim,
    xlabel="Period (s)",ylabel="pixels / (m/s)")    
plot!(hpt2,1 ./TXFRF[pltidx],TXFRD_smth[pltidx],lc=:darkred,label="",)
hpt2all = plot(hpt2,hpt2l,layout=grid(1,2),size=(1000,400),bottom_margin=5mm)
# save figure
savefig(hpt2all,string(c_dataout,c_runname,"txfr2.pdf"))
# and the weights as well
hpwght = plot(1 ./TXFRF,map(x->median(filter(!isnan,TXFRMwght[x,:])),1:lastindex(TXFRF)),
    label="",xlabel="Period (s)",title="Weights by Period",xlim=plotxlim,lc=:black,xminorgrid=true)
savefig(hpwght,string(c_dataout,c_runname,"wghts.pdf"))
hpwght = plot(1 ./TXFRF,map(x->median(filter(!isnan,TXFRMwght[x,:])),1:lastindex(TXFRF)),
    label="",xlabel="Period (s)",title="Weights by Period",xlim=plotxlim,lc=:black,
    xaxis=:log,xminorgrid=true)
savefig(hpwght,string(c_dataout,c_runname,"wghtsl.pdf"))

## DO THE THEORETICAL TRANSFER FUNCTION
Trange = 0.01:0.01:100 # seconds
Frange = 1 ./Trange
w = 2*pi./Trange
wtxfr = 2*pi./(1 ./TXFRF)
w0 = 2*pi/T0
wg = 2*pi/Tg
if dampingcritical
    Q = (w.^3)./((w0^2 .+ w.^2).*(wg^2 .+ w.^2)) # EQ 30 benioff 1932
    Qtxfr = (wtxfr.^3)./((w0^2 .+ wtxfr.^2).*(wg^2 .+ wtxfr.^2))
else
    Q = (w.^3)./((w0^2 .+ w.^2).*sqrt.((wg^2 .- w.^2).^2 .+ 4*(eg^2)*w.^2)) # EQ 27 benioff 1932
    Qtxfr = (wtxfr.^3)./((w0^2 .+ wtxfr.^2).*sqrt.((wg^2 .- wtxfr.^2).^2 .+ 4*(eg^2)*wtxfr.^2))
end
# plot empirical transfer function and theoretical one
fidx = findall(minimum(Frange).<=TXFRF.<=maximum(Frange))
hptve = plot(1 ./TXFRF[fidx],TXFRD[fidx],label="",title="Theoretical vs Empirical 1-100s",
    xminorgrid=true,lc=:black,xlabel="Period (s)",ylabel="pixels / (m/s)") 
ax2tve = twinx(hptve)
plot!(ax2tve,Trange,Q,lc=:blue,lw=1.5,label="Theoretical",
    y_guidefontcolor=:blue,y_foreground_color_axis=:blue,
    y_foreground_color_text=:blue,y_foreground_color_border=:blue,)
hptvelog = deepcopy(hptve)
plot!(hptvelog,xaxis=:log)
# 20 second version
fidx = findall((1/20).<=TXFRF.<=maximum(Frange))
hptve20 = plot(1 ./TXFRF[fidx],TXFRD[fidx],label="",title="Theoretical vs Empirical 1-20s",
    xminorgrid=true,lc=:black,xlabel="Period (s)",ylabel="pixels / (m/s)") 
ax2tve20 = twinx(hptve20)
fidx2 = findall(Trange.<=20)
plot!(ax2tve20,Trange[fidx2],Q[fidx2],lc=:blue,lw=1.5,label="Theoretical",
    y_guidefontcolor=:blue,y_foreground_color_axis=:blue,
    y_foreground_color_text=:blue,y_foreground_color_border=:blue,)
hptvelog20 = deepcopy(hptve20)
plot!(hptvelog20,xaxis=:log)
# aggregate
hptveall = plot(hptve,hptve20,hptvelog,hptvelog20,layout=grid(2,2),size=(1400,1000))
savefig(hptveall,string(c_dataout,c_runname,"theoretical_v_empirical.pdf"))
# plot just theoretical
hptheor = plot(1 ./Frange[2:end],Q[2:end],xlabel="Period (s)",ylabel="Q",
    xaxis=:log,minorgrid=true,lc=:black,label="",title=string("T0=",T0," Tg=",Tg))
savefig(hptheor,string(c_dataout,c_runname,"theoretical.pdf"))
hptheorl = plot(1 ./Frange[2:end],Q[2:end],xlabel="Period (s)",ylabel="Q",xlim=(0,32),
    yaxis=:log,minorgrid=true,lc=:black,label="",title=string("T0=",T0," Tg=",Tg))
savefig(hptheorl,string(c_dataout,c_runname,"theoretical_log_benioff.pdf"))

## FIT Q AGAINST TXFRD
if fullfit
    A = [reshape(ones(size(TXFRM)).*Qtxfr,length(TXFRM),1) ones(length(TXFRM),1)]
    b = vec(reshape(TXFRM,length(TXFRM),1))
    ftmp = vec(reshape(ones(size(TXFRM)).*TXFRF,length(TXFRM),1))
    if useweights
        wghts = vec(reshape(TXFRMwght,length(TXFRMwght),1))
    else
        wghts = vec(ones(length(TXFRMwght),1))
    end
else
    A = [Qtxfr ones(length(Qtxfr),1)]
    b = TXFRD_smth
    ftmp = deepcopy(TXFRF)
    if useweights
        wghts = map(x->median(filter(!isnan,TXFRMwght[x,:])),1:lastindex(TXFRF))
    else
        wghts = vec(ones(length(TXFRF,1)))
    end
end
gidx = findall(.!isnan.(b))
gidx2 = findall(plims[1] .<= 1 ./ftmp[gidx] .<= plims[2])
x = RobustLeastSquares.solve(A[gidx[gidx2],:],b[gidx[gidx2]],wghts[gidx[gidx2]],:qr)
# add to plots
TXFRQ = Qtxfr.*x[1] .+ x[2]
hpfit = scatter(A[gidx[gidx2],1],b[gidx[gidx2]],zcolor=1 ./ftmp[gidx[gidx2]],label="",
    xlabel="Theoretical",ylabel="Empirical",colorbar_title="\nPeriod (s)",right_margin=10mm)
plot!(hpfit,A[gidx[gidx2],1],A[gidx[gidx2],1].*x[1] .+ x[2],
    label=string("m=",x[1],", b=",x[2]),legend=:outerbottom)
savefig(hpfit,string(c_dataout,c_runname,"theoretical_fit.pdf"))
#plot(1 ./TXFRF[2:end],TXFRQ[2:end])
plot!(hpt,1 ./TXFRF[2:end],TXFRQ[2:end],lw=2,label="Theoretical Fit",)
plot!(hptg,1 ./TXFRF[2:end],log10.(TXFRQ[2:end]),lw=2,label="Theoretical Fit",)
plot!(hptgl,log10.(1 ./TXFRF[2:end]),log10.(TXFRQ[2:end]),lw=2,label="Theoretical Fit",)
plot!(hpt2,1 ./TXFRF[pltidx],TXFRQ[pltidx],lw=2,label="Theoretical Fit",)
plot!(hpt2l,1 ./TXFRF[pltidx],TXFRQ[pltidx],lw=2,label="Theoretical Fit",)
hpt2all = plot(hpt2,hpt2l,layout=grid(1,2),size=(1000,400),bottom_margin=5mm)
# save figures again
savefig(hpt,string(c_dataout,c_runname,"txfr_fit.pdf"))
savefig(hptg,string(c_dataout,c_runname,"txfrgrd_fit.pdf"))
savefig(hptgl,string(c_dataout,c_runname,"txfrgrdl_fit.pdf"))
savefig(hpt2all,string(c_dataout,c_runname,"txfr2_fit.pdf"))

## WRITE OUT DATA
save(string(c_dataout,c_runname,"txfr.jld"),
    "freq",TXFRF,
    "txfr",TXFRD,
    "txfrQ",TXFRQ,
    "txfr5",TXFRD5,
    "txfr95",TXFRD95,
)

# finish
print("Done!\n")