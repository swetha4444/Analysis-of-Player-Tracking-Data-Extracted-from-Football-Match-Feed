import pandas as pd
import numpy as np
import DecisionMaking.Metrica_IO as mio
import DecisionMaking.Metrica_Viz as mviz
import DecisionMaking.Metrica_Velocities as mvel
import DecisionMaking.Metrica_PitchControl as mpc
import numpy as np
import pandas as pd
import DecisionMaking.Metrica_EPV as mepv
import streamlit as st

def app():
   try: 

    event_data = st.sidebar.file_uploader("Upload event data")
    if event_data is not None:
            data = pd.read_csv(event_data)
        #st.write(df)

    


    home_tracking = st.sidebar.file_uploader("Upload home tracking")
    if home_tracking is not None:
            tracking_data = pd.read_csv(home_tracking)
        #st.write(df1)
        
    away_tracking = st.sidebar.file_uploader("Upload away tracking")
    if away_tracking is not None:
            tracking1 = pd.read_csv(away_tracking)
    


        

    data = mio.to_metric_coordinates(data)
    events = data
    
    data = data[data['Type']=='PASS']
    data = data[data['Team']=='Home']

    player = data['From'].tolist()
    player = np.array(player)
    player = np.unique(player)
    
    
    
    
    player[0],player[1] = player[1],player[0]
    player = st.selectbox('Choose player to analyse',player)
    
    event_data = data[data['From']==player]

    
    tracking_home = tracking_data
    tracking_away = tracking1
    
    params = mpc.default_model_params()
# find goalkeepers for offside calculation
    GK_numbers = [mio.find_goalkeeper(tracking_home),mio.find_goalkeeper(tracking_away)]



    home_attack_direction = mio.find_playing_direction(tracking_home,'Home') # 1 if shooting left-right, else -1
    
    

    frame = event_data['Start Frame']

    frame = frame.reset_index()
    tracking=[]

    for i in range(0,len(frame)):
        finalf = tracking_home[tracking_home['Frame']==frame['Start Frame'][i]]
        tracking.append(finalf)
    
    tracking = pd.concat(tracking)

    final_frame = tracking['Frame'].tolist()





    EPV = pd.read_excel("xT.xlsx", header=None)
    #EPV = pd.read_csv("EPV_grid.csv")

#plot the EPV surface
    mviz.plot_EPV(EPV,field_dimen=(106.0,68),attack_direction=home_attack_direction)
    
    no_of_pass = len(frame)
    options = []
    
    for i in range(0,no_of_pass):
        options.append(i+1)
    
    pass_analyse = st.selectbox("Choose which pass to analyse",options)
    
    pass_analyse = pass_analyse - 1
    
    event_number = frame['index'][pass_analyse]
    
    
    EEPV_added, EPV_diff = mepv.calculate_epv_added( event_number, events, tracking_home, tracking_away, GK_numbers, EPV, params)
    PPCF,xgrid,ygrid = mpc.generate_pitch_control_for_event(event_number, events, tracking_home, tracking_away, params, GK_numbers, field_dimen = (106.,68.,), n_grid_cells_x = 16, offsides=True)
    grid, fig,ax = mviz.plot_EPV_for_event( event_number, events,  tracking_home, tracking_away, PPCF, EPV, annotate=True, autoscale=True)
    
    st.write(fig)

#fig.suptitle('Pass value added: %1.3f' % EEPV_added, y=0.95 )
    #mviz.plot_pitchcontrol_for_event(event_number, events,  tracking_home, tracking_away, PPCF, annotate=True)


    pass_frame = events.loc[event_number]['Start Frame']
    pass_frame = pass_frame - 1
    player_loc_frame = tracking_home.loc[pass_frame]




    alist=[]

    for i in range(1,15):
        x = "Home_" + str(i) + "_x"
        y = "Home_" + str(i) + "_y"
        player_id = 'Player' + str(i)
        period = player_loc_frame['Period']
        time = player_loc_frame['Time [s]']
        X = player_loc_frame[x]
        Y = player_loc_frame[y]
    
    
        d = [period,time,player_id,X,Y]
    
        alist.append(d)
    
    df = pd.DataFrame(alist, columns = ['Period', 'Time','ID','x','y'])


    xT = np.array(grid)

    xT_rows, xT_cols = grid.shape

    df['x1_bin'] = pd.cut(df['x'], bins=xT_cols, labels=False)
    df['y1_bin'] = pd.cut(df['y'], bins=xT_rows, labels=False)

    df = df.dropna()
    df['x1_bin'] = df['x1_bin'].astype(int)
    df['y1_bin'] = df['y1_bin'].astype(int)

    df['end_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)


    df_event = events[events['Start Time [s]']==time]

    df['start_x'] = df_event["Start X"][event_number]
    df['start_y'] = df_event['Start Y'][event_number]
    df['Passed to'] = df_event['To'][event_number]
    df['Passed From'] = df_event['From'][event_number]

    df['x2_bin'] = pd.cut(df['start_x'], bins=xT_cols, labels=False)
    df['y2_bin'] = pd.cut(df['start_y'], bins=xT_rows, labels=False)
    df['start_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)


    df['Pass value'] = df['end_zone_value'] - df['start_zone_value']

    df['increases chances,%'] = df['Pass value']*100
    for i in range(1,len(df)):
        if df['ID'][i]==df['Passed From'][i]:
            df = df.drop(df.index[i])




    df.reset_index(inplace=True)
    max_value = max(df['Pass value'])

    for i in range(1,len(df)):
        if df['ID'][i]==df['Passed to'][i]:
            attempted_value = df['Pass value'][i]
            
    answer = (attempted_value/max_value)*100
    
    answer = str(round(answer, 2))

    answer = str(answer) + "%"

   
    st.write("Players decision making ability in current frame: ",answer)
    
    


        
   except:
       st.success("You've Successfully Logged in")


def goto():
    st.title("Quantifying decision making ability of a pass")
    app()
    


        
    if st.sidebar.button('Sign out'):
        st.session_state['valid_user'] = False
        st.experimental_rerun()
        
        

