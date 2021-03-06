if 1:
    #Allocate space for effective wetness/dryness

    Ud = Z*0.
    Uw = Z*0.
    X1 = Z*0.
    X2 = Z*0.
    X3 = Z*0.
    Pe = Z*0.
    BT = Z*0.
    montho = MV.zeros(Z.shape) #log how many months wet/dry spells have been going on
    X = MV.zeros(Z.shape)
    
    #Start in month 1, inheriting no wetness/dryness from previous month
    i = 0
    XX = Z[i]/3. #PDSI value if there is no wet or dry spell underway
    X1i = MV.where(XX>0,XX,0)
    X1i = MV.where(XX>1,0,X1i)

    X2i = MV.where(XX<0,XX,0)
    X2i = MV.where(XX<-1,0,X2i)

    X3i = MV.where(MV.absolute(XX)>1,XX,0)
    #how many months has a wet/dry spell lasted?

    nmonths = MV.where(MV.absolute(XX)>1,1,0)
    montho[i] = nmonths
    X[i]=XX
    X1[i]=X1i
    X2[i]=X2i
    X3[i]=X3i

   
    for i in range(Z.shape[0])[1:6]: #Loop over time
    if 1:
        XX=Z[i]/3.

        nmonths = montho[i-1] #months so far in dry spell
        #These value will be used when calculating the probability of ending a wet/dry spell.
        #An established spell will be maintained at |PDSI|>=0.5 with a value of |Z| >=0.15
        Ud[i] = Z[i]-0.15 
        Uw[i] = Z[i] + 0.15

        this_month_wet = XX>0
        this_month_dry = XX<0
        this_month_neutral = XX==0

        #PDSI value if initiating a dry spell
        X2i = MV.where((X2[i-1]*(1-0.103)+XX)<XX,X2[i-1]*(1-0.103)+XX,XX)
        X1i =  MV.where((X1[i-1]*(1-0.103)+XX)>XX,X1[i-1]*(1-0.103)+XX,XX)
                     
       

        transitional_value = MV.where(this_month_wet,X1[i],0)
        transitional_value = MV.where(this_month_dry,X2[i],0)
        extreme_transitional_value = MV.absolute(transitional_value)>1

        
        
        last_month_neutral = X3[i-1] ==0
        last_month_dry = X3[i-1]<0
        last_month_wet = X3[i-1]>0

    
        #Moisture needed to end drought or wet spell:
        Ze_drought = -2.691*X3[i-1]-1.5
        Ze_wet_spell = -2.691*X3[i-1]+1.5
        Ze = MV.where(last_month_dry,Ze_drought,0)
        Ze = MV.where(last_month_wet,Ze_wet_spell,Ze)

        #Probability that last month's wet or dry spell will end
        P_end_wet = end_spell_P(Ud,Ze_wet_spell,nmonths,i,typ="wet")
        P_end_dry = end_spell_P(Uw,Ze_drought,nmonths,i,typ="dry")
        P_end_spell = MV.where(last_month_dry,P_end_dry,0)
        P_end_spell = MV.where(last_month_wet,P_end_wet,P_end_spell)
        Pe[i]=P_end_spell
   
        # If this was a wet month
        condition = this_month_wet
        #If last month was in a dry spell
        condition1 = np.ma.logical_and(condition,last_month_dry)
        #If the dry period is definitely ending
        dry_definitely_ending = P_end_dry >=100
        condition2 = np.ma.logical_and(condition1,dry_definitely_ending)
        #PDSI = wet transitional value
        Xi=MV.where(condition2,X1i,0)
        #  If wet transitional value indicates very wet conditions
        very_wet=X1i>1
        # set X3 to wet transitional value
        condition3=np.logical_and(condition2,very_wet)
        X3i=MV.where(condition3,X1i,0)
        
        X1i=MV.where(condition3,0,X1i)
        #Log beginning of wet spell
        nmonths = MV.where(condition3,1,0)

        #If the dry period is definitely NOT ending
        dry_definitely_NOT_ending = P_end_dry <= 0
        condition2 = np.ma.logical_and(condition1,dry_definitely_NOT_ending)
        X2i=MV.where(condition2,0,X2i)

        #If last month was in a wet spell
        condition1 = np.ma.logical_and(condition,last_month_wet)
        #If the wet period is definitely ending
        wet_definitely_ending = P_end_wet >=100
        condition2 = np.ma.logical_and(condition1,wet_definitely_ending)
        #PDSI = dry transitional value
        Xi=MV.where(condition2,X2i,Xi)

         #  If dry transitional value indicates very dry conditions
        very_dry=X2i<=-1
        # set X3 to dry transitional value
        condition3=np.logical_and(condition2,very_dry)
        X3i=MV.where(condition3,X2i,X3i)
        
        X2i=MV.where(condition3,0,X1i)
        #Log beginning of dry spell
        nmonths = MV.where(condition3,1,nmonths)

        #If the wet spell is definitely NOT ending
        wet_definitely_NOT_ending = P_end_wet <= 0
        condition2 = np.ma.logical_and(condition1,wet_definitely_NOT_ending)
        X1i=MV.where(condition2,0,X1i)

        #if last month was neutral but this month has a positive Z:
        condition1 = np.ma.logical_and(condition,last_month_neutral)
        wet_spell_establishing = X1i>0.5
        condition2=np.logical_and(condition1,wet_spell_establishing)
        #PSDI = wet transitional value
        Xi=MV.where(condition2, X1i,Xi)
        #Begin wet spell
        nmonths=MV.where(condition2,1,nmonths)
        #if wet spell is atarting off strong
        wet_strong = X1i>=1
        condition3=np.logical_and(condition2,wet_strong)
        #Set wet persistence value to wet transitional value
        X3i=MV.where(condition3,X1i,X3i)
        X1i=MV.where(condition3,0,X1i)

        #If this was a dry month
        condition = this_month_dry
         #If last month was in a wet spell
        condition1 = np.ma.logical_and(condition,last_month_wet)
        #If the wet period is definitely ending
        wet_definitely_ending = P_end_wet >=100
        condition2 = np.ma.logical_and(condition1,wet_definitely_ending)
        #PDSI = dry transitional value
        Xi=MV.where(condition2,X2i,Xi)
        #  If dry transitional value indicates very dry conditions
        very_dry=X2i<=-1
        # set X3 to dry transitional value
        condition3=np.logical_and(condition2,very_dry)
        X3i=MV.where(condition3,X2i,X3i)
        
        X2i=MV.where(condition3,0,X1i)
        #Log beginning of dry spell
        nmonths = MV.where(condition3,1,nmonths)

        #If the wet spell is definitely NOT ending
        wet_definitely_NOT_ending = P_end_wet <= 0
        condition2 = np.ma.logical_and(condition1,wet_definitely_NOT_ending)
        X1i=MV.where(condition2,0,X1i)

        #If last month was in a dry spell
        condition1 = np.ma.logical_and(condition,last_month_dry)
        #If the dry period is definitely ending
        dry_definitely_ending = P_end_dry >=100
        condition2 = np.ma.logical_and(condition1,dry_definitely_ending)
        #PDSI = wet transitional value
        Xi=MV.where(condition2,X1i,0)
        #  If wet transitional value indicates very wet conditions
        very_wet=X1i>1
        # set X3 to wet transitional value
        condition3=np.logical_and(condition2,very_wet)
        X3i=MV.where(condition3,X1i,0)
        
        X1i=MV.where(condition3,0,X1i)
        #Log beginning of wet spell
        nmonths = MV.where(condition3,1,0)

        #If the dry period is definitely NOT ending
        dry_definitely_NOT_ending = P_end_dry <= 0
        condition2 = np.ma.logical_and(condition1,dry_definitely_NOT_ending)
        X2i=MV.where(condition2,0,X2i)

         #if last month was neutral but this month has a negative Z:
        condition1 = np.ma.logical_and(condition,last_month_neutral)
        dry_spell_establishing = X2i<=0.5
        condition2=np.logical_and(condition1,dry_spell_establishing)
        #PSDI = dry transitional value
        Xi=MV.where(condition2, X2i,Xi)
        #Begin dry spell
        nmonths=MV.where(condition2,1,nmonths)
        #if dry spell is atarting off strong
        dry_strong = X2i<=-1
        condition3=np.logical_and(condition2,dry_strong)
        #Set dry persistence value to dry transitional value
        X3i=MV.where(condition3,X2i,X3i)
        X2i=MV.where(condition3,0,X2i)

        X1i=MV.where(X1i<0,0,X1i)
        X2i = MV.where(X2i>0,0,X2i)
    
        #if the probability the spell has ended is not 100% and it wasn't a neutral month
        condition = P_end_spell<100
        X3i = MV.where(condition,X3[i-1]*.897+XX,X3i)
        #If the sign of X3 didn't change over the time step
        sign_same = np.sign(X3[i-i]) == np.sign(X3i)
        condition1=np.ma.logical_and(condition,sign_same)
        #If X3 is still extreme
        extreme = MV.absolute(X3i)>0.5
        condition2=np.ma.logical_and(condition1,extreme)
        #The spell continues
        nmonths=MV.where(condition2,nmonths+1,nmonths)
        #Otherwise, the spell ends
        condition2=np.ma.logical_and(condition1,~extreme)
        nmonths=MV.where(condition2,0,nmonths)
        #If the sign did change
        sign_different = ~sign_same
        condition1=np.ma.logical_and(condition,sign_different)
        #IF the X3 anomaly is large
        very_extreme = MV.absolute(X3i)>=1
        condition2=np.ma.logical_and(condition1,very_extreme)
        #new dry or wet spell begins
        nmonths=MV.where(condition2,1,nmonths)
        #otherwise neutral conditions continue
        condition2=np.ma.logical_and(condition1,~very_extreme)
        nmonths=MV.where(condition2,0,nmonths)
    
        montho[i]=nmonths
        #Decide what PDSI is
        #If there is no event established or in progress
        no_event = nmonths ==0
        tending_toward_wet = X1i>-X2i
        wet_condition = np.logical_and(no_event,tending_toward_wet)
        Xi=MV.where(wet_condition,X1i,Xi)
        tending_toward_dry = X1i<=-X2i
        dry_condition = np.logical_and(no_event,tending_toward_dry)
        Xi=MV.where(dry_condition,X2i,Xi)

        #If a wet or dry spell has begun
        event_begun = nmonths == 1
        began_last_month = X3i==0
        condition = np.logical_and(event_begun,began_last_month)
        condition1= np.logical_and(condition,tending_toward_wet)
        Xi = MV.where(condition1,X1i,Xi)
        condition1= np.logical_and(condition,tending_toward_dry)
        Xi = MV.where(condition1,X2i,Xi)
        began_this_month = X3i !=0
        condition = np.logical_and(event_begun,began_this_month)
        pdsi_not_assigned = Xi==0
        Xi=MV.where(np.logical_and(pdsi_not_assigned,condition),X3i,Xi)
        #If beyond the first month of a wet or dry spell
        Xi=MV.where(nmonths>1,X3i,Xi)
        X[i]=Xi
        X1[i]=X1i
        X2[i]=X2i
        X3[i]=X3i
        
        #BACKTRACK
        ending_or_confirmed = np.logical_or(P_end_spell>=100,P_end_spell<=0)
        condition = np.logical_and(montho[i]==1,ending_or_confirmed)
        
        
        s_orig = i
        #Find the month when the previous wet or dry spell ended
        
        Pe_trunc = Pe[:s_orig]
        months_previous = Pe_trunc.shape[0]
        Pe_relevant = MV.where(np.repeat(condition.asma()[np.newaxis],months_previous,axis=0),Pe_trunc,0)
        
        #HACKY PYTHON REPLACEMENT FOR MATLAB FIND LAST
        I,J,K=np.ma.where(Pe_relevant>=100)

        datapoints = zip(J,K) #spatial grid points at which the previous wet or dry spell ended
        #need to find the most recent month of end for each grid point
        stringdata=[str(x) for x in datapoints]
        unique_points = np.unique(stringdata)        
        
        
         
        for point in unique_points:
            lat,lon=ast.literal_eval(point)
            index=np.max(np.where(np.array(stringdata)==point)[0]) #find the last place at which the point occurs
            s=I[index]
            #Find months prior to the 100% transition where the probability of the termination of the previous spell was above the BTthresh probability
       
            r=s
            if (Pe[s-1,lat,lon]>0 and Pe[s-1,lat,lon]<100):
                for c in np.arange(1,s)[::-1]:
                    if (Pe[c,lat,lon] > BTthresh and Pe[c,lat,lon]< 100):
                        r=c
                    else:
                        break
            #extend backtracking period forward from the month when  the dry or wet spell had a 100% of terminating to the month
            #when the next dry or wet spell officially began
            F = np.arange(r,s)
            if len(F)==0:
                F=np.array([r])
            if s<s_orig:
                F=np.arange(F[0],s_orig+1)
            #determine whether backtracking is occuring as the result of the initiation of a wet or dry spell
            
            m = np.min(np.where(montho[s:s_orig+1,lat,lon]==1)[0])
            if X[s+m,lat,lon]>0:
                pon=0 #wet spell
            else:
                pon=1 #dry spell
            #Backtrack and assign new PDSI values
            possible_PDSI = [X1[F,lat,lon][0],X2[F,lat,lon][0]]
            the_month = len(F)-1
            
            last_month_pon=0
            ponflip=lambda thing: 1 if thing is 0 else 0
            while the_month >= 0:
                        
            
                #if next month is in a dry or wet phase, assign this month the transitional PDSI
                # value associated with that phase as long as it is non-zero
                if possible_PDSI[pon][the_month] !=0:
                    X[F[the_month],lat,lon] = possible_PDSI[pon][the_month]
                else:
                    #if transitional value for next month's phase is zero, assign
                    # this month to the other transitional phase
                     X[F[the_month],lat,lon] = possible_PDSI[ponflip(pon)][the_month]
                     pon=ponflip(pon)
                #if both transitional PDSI values were zero
                if X[F[the_month],lat,lon] ==0:
                    the_X3 = X3[F[the_month],lat,lon]
                    t=np.sign(the_X3)
                    especially_strong = np.abs(the_X3)>=1
                    consistent_with_next_month_phase= (1-t)/2 == last_month_pon
                    if (especially_strong or consistent_with_next_month_phase):
                        #Assign X3 as this month's PDSI value
                        X[F[the_month],lat,lon] = the_X3
                        if  X[F[the_month],lat,lon] >0:
                            pon=0
                        else:
                            pon=1
                    #if X3 doesn't satisfy the above conditions, leave PDSI =0
                    else:
                        if last_month_pon !=0:
                            pon = last_month_pon #assign this month to the dry or wet phase that eventually develops
                            # indicate whether backtracking assigned this month to a wet or dry phase
                    BT[F[the_month]]=pon
                    last_month_pon=pon
                    the_month=the_month-1
                    
                #recalculate running tally of months in a dry or wet spell
        
        

            nmonths[lat,lon] = montho[F[0]-1,lat,lon]
            if ((the_month < len(F)-1) and (F[-1] < Z.shape[0])):
                F=np.append(F[the_month:],F[-1]+1)
                for newj in range(len(F)):
                    if np.abs(X[F[newj],lat,lon])>1:
                        if np.sign(X[F[newj],lat,lon]) == np.sign(X[F[newj]-1,lat,lon]):
                            nmonths[lat,lon]=nmonths[lat,lon]+1
                        else:
                            nmonths[lat,lon]=1
                    else:
                        nmonths[lat,lon]=0
                    montho[F[newj],lat,lon]=nmonths[lat,lon]
    PDSI=X
    PDSI.id="pdsi"
    PDSI.setAxisList(Z.getAxisList())
