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
c_runname = "TEST6.0/" # make sure to add '/' to get folder
# ISC data files
c_old_ISC = string(usr_str,"Research/FindEQDoublets/ISC_M6_1936_1941.txt")
c_new_ISC = string(usr_str,"Research/FindEQDoublets/ISC_M6_1988_2024.txt")
# HRV data files
c_HRV_old = string(usr_str,"Downloads/HRV_SAC_ANALOG/LPZ/")
c_HRV_save = string(c_dataout,"HRV_1936_1940_LPZ.jld")
# HRV manual gain correction
station_gains_file = [] # use this empty to avoid correcting gains
station_gains_file = string(usr_str,"Research/HRV_BHZ_Gain.txt")
# EQ save files
c_oldEQ_save = string(c_dataout,c_runname,"HRV_1936_1940_LPZ_oldEQ.jld")
c_newEQ_save = string(c_dataout,c_runname,"HRV_1988_2023_BHZ_newEQ.jld")
# search parameters
deplim = 50 # deepest limit for depth (km)
magmin = 6.0 # smallest allowable mag
distdiff = 50 # allowable distance difference (km)
depdiff = 25 # allowable depth difference (km)
magdiff = 0.2
surfvel = 3.33 # surface wave velocity km/s
oldEQmanualcheck = false # run GUI to check waveforms
oldEQignore = [] # matchID strings to ignore
windowstart = -Dates.Minute(15)
windowend = Dates.Minute(45) # window for surface waves
datathresh = 0.9 # data coverage required in window
hrv_lat = 42.5060
hrv_lon = -71.5580

## SETUP RUNNAME DIR
if !isdir(string(c_dataout,c_runname))
    mkdir(string(c_dataout,c_runname))
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
    tmpvar = []
else # build old EQ events
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
                        plot!(hps,1 ./specttmpF[2:end],movmean(specttmpPSD[2:end],50),lc=:red,label="smoothed")
                            # smoothing is roughly a 1Hz window
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
        #print(string(il,"\n"))
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
    for i in ProgressBar(1:lastindex(newEQtme))
        #print(string("i=",i,"\n")) # error on 861
        # check depths
        oldidx = findall(newEQdep[i]-depdiff .<= oldEQdep .<= newEQdep[i]+depdiff)
        # check proximity 
        dtmp = lf.gcdist(newEQlon[i],newEQlat[i],oldEQlon[oldidx],oldEQlat[oldidx],Ga)
        dtmp = dtmp./1000 # m to km
        oldidx2 = oldidx[findall(dtmp.<=distdiff)]
        # check magnitude
        oldidx3 = oldidx2[findall(newEQmag[i]-magdiff .<= oldEQmag[oldidx2] .<= newEQmag[i]+magdiff)]
        if !isempty(oldidx3)
            # print(string("i=",i,"\n"))
            # get match info
            matchID = map(x->string(
                    Dates.format(oldEQtme[oldidx3[x]],"yyyymmdd_HHMM"),
                    "_M",oldEQmag[oldidx2[x]]),
                1:lastindex(oldidx2))
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
            statmp = get_stations(network="IU", station="HRV", channel="BHZ",
                starttime=Dates.format(stime,"yyyy-mm-ddTHH:MM:SS"), 
                endtime=Dates.format(etime,"yyyy-mm-ddTHH:MM:SS"))
            if length(statmp)>1 # specify specifically 00 if available
                statmp = get_stations(network="IU", station="HRV", channel="BHZ", location="00",
                    starttime=Dates.format(stime,"yyyy-mm-ddTHH:MM:SS"), 
                    endtime=Dates.format(etime,"yyyy-mm-ddTHH:MM:SS"))
            end
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
                # read gain (assuming flat response, not a bad one for BB HRV)
                if !isempty(station_gains_file)
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
                # subtract non-paramteric trend
                tracetmp = tracetmp .- movmean(tracetmp,convert(Int,round(length(tracetmp)/10)))
                # calculate fft
                noNaNtrace = deepcopy(tracetmp)
                noNaNtrace[isnan.(noNaNtrace)] .= mean(filter(!isnan,noNaNtrace))
                specttmpD = FFTW.rfft(noNaNtrace)
                specttmpF = FFTW.rfftfreq(length(noNaNtrace),1/(Dates.value(samprate)/1000))
                # convert to PSD
                specttmpPSD = 2*(1/((Dates.value(samprate)/1000)*length(noNaNtrace))).*(abs.(specttmpD).^2)
                # plot and write out waveform and spectras
                hpw = plot(traceTtmp,tracetmp,lc=:black,legend=false,ylabel="pixels",
                    title=string(Dates.format(newEQtme[i],"yyyy-mm-ddTHH:MM:SS.sss"),
                        "; M",newEQmag[i],"; (",newEQlat[i],", ",newEQlon[i],", ",newEQdep[i],
                        "); ",round(dHRV),"km (",round(dHRV/111.1,digits=2),"deg) -> ",ttime/1000,"s"))
                hps = plot(1 ./specttmpF[2:end],specttmpPSD[2:end],xaxis=:log,yaxis=:log,lc=:black,
                    label="raw",xlabel="Period (s)",ylabel="pixels^2/Hz",minorgrid=true) 
                plot!(hps,1 ./specttmpF[2:end],movmean(specttmpPSD[2:end],50),lc=:red,label="smoothed")
                    # smoothing is roughly a 1Hz window
                hpall = plot(hpw,hps,layout=grid(2,1),size=(1000,1000))
                # save data and plot
                ID = string(Dates.format(newEQtme[i],"yyyymmdd_HHMM"),"_M",newEQmag[i])
                savefig(hpall,string(c_dataout,c_runname,"newevents/",ID,".pdf"))
                push!(newEQtrace,tracetmp)
                push!(newEQtraceT,tracetmp)
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

## DO THE COMPARISON FOR EXISTING MATCHES