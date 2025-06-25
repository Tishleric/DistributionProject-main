# External Libraries: Selenium, BeautifulSoup, pandas   
# Internal Libraries: os,time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import traceback as trb
#from returns_main import input_folder

def StartWebscrapper(**kwargs): 
    """
    Runner function for ChangeTimezone and PrepareCalendar functions
    """

    MyTargetTimezone=kwargs.get('TargetTimezoneValue')
    all_timezones=kwargs.get('all_timezones')
    # Call the function to change the timezone
    (default, current)= ChangeTimezone(**kwargs)

    # Check if the timezone was updated successfully
    if current != MyTargetTimezone:
        print(f'Unable to change timezone. Data will be shown as per {default}')
        try:
            kwargs['TargetTimezoneName']=all_timezones[default]
        except:
            kwargs['TargetTimezoneName']=="UnavailableTimeZone"

    MaxAttempts=3
    for _ in range(MaxAttempts):
        try:
            return PrepareCalendar(**kwargs)
        except Exception as e:
            print('Some error occurred in fetching data. Retrying...')
            print(e)
            trb.print_exc()
            continue


def ChangeTimezone(**kwargs):  # Default value for the timezone
    """
    Select the Target Timezone from Trading Economics Website
    """
    driver=kwargs.get('driver')
    NewTimezone=kwargs.get('TargetTimezoneValue')

    print('Starting....')
    default=""
    current=""
    MaxAttempts = 20
    for _ in range(MaxAttempts):
        try:
            dropdown_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "DropDownListTimezone"))
            )
            time.sleep(5)
            select = Select(dropdown_element)
            
            # Get the currently selected option
            selected_option = select.first_selected_option
            current = selected_option.text  
       
            
            if current == NewTimezone:
                print(f"Timezone successfully updated from {default} to {NewTimezone}")
                break  # Exit if the timezone is already set to the desired one
            else:
                default=current

            select.select_by_visible_text(NewTimezone)

            # Wait briefly for the dropdown to update and get the new selected option
            WebDriverWait(driver, 2).until(
                EC.staleness_of(selected_option)  # Wait until the previous element becomes stale
            )
            time.sleep(5)
            selected_option = select.first_selected_option
            current = selected_option.text  

            if current == NewTimezone:
                print(f"Timezone successfully updated from {default} to {NewTimezone}")
                break 

        except Exception as e :

            if 'stale' in str(e):
                 continue
            else:
                 print(e)
                 break
    return (default, current)


def PrepareCalendar(**kwargs):
        
        """
        Preparing the calendar out of data colected from Trading Economics 
        """

        store_dic=GetTiers(**kwargs)
        kwargs['store_dic']=store_dic
        table_data=GetValues(**kwargs)
      
        mydf=pd.DataFrame(table_data)

        mydf.insert(1,'Events',mydf[4])
        mydf.rename(columns={0:'Date',1:'Actual',2:'Previous',3:'Consensus',4:'Forecast',5:'Tier'},inplace=True)
    
        # Replace 'Forecast' in 'Events' and set other columns to ""
        cols_to_replace = mydf.columns[1:]  # Select all columns except the 1th one
        mydf.loc[mydf['Events'] == 'Forecast', cols_to_replace] = ""

        mydf.insert(1,'Country',mydf['Actual'])
        mydf['Actual']=mydf['Tier']
        mydf['Previous']=mydf[6]
        mydf['Consensus']=mydf[7]
        mydf['Forecast']=mydf[8]
        mydf['Tier']=mydf[11]
        mydf.drop(axis=1,inplace=True,columns=[6,7,8,9,10,11])
        mydf=mydf.drop_duplicates().reset_index(drop=True)

        mydf=mydf[mydf.isnull()==False]
        mydf.dropna(inplace=True)
        mydf = mydf.replace('®', '', regex=True)#regex replaces even if the sign is inside the text and not just the sign.
        mydf['Tier']=mydf['Tier'].mask(mydf['Tier']=='N/A').ffill() #If tier is N/A, then replace it with not null value that came immediately before that N/A cell.
        #mydf=AddTimestamp(mydf.copy())
        return StoreCalendar(mydf.copy(),**kwargs)


def GetTiers(**kwargs):
    """
    Return a dictionary with details of Tier for different events. 
    Tiers can be 1,2,3 and denote significance of event, 1 being highest.
    """
    time.sleep(10)
    driver=kwargs.get('driver')
    #Get type of font from each element. Find all elements within the table
    elements = driver.find_elements(By.TAG_NAME, "SPAN")#//*[contains(@id, 'calendar')]//tr//td")  # Adjust as needed
    store_dic={} #Store type of font i.e helps get tier data for the events. Used later in Part2
    for _, element in enumerate(elements):
        # Retrieve and store the computed styles
        font_weight = driver.execute_script("return window.getComputedStyle(arguments[0], null).getPropertyValue('font-weight');", element)
        color = driver.execute_script("return window.getComputedStyle(arguments[0], null).getPropertyValue('color');", element)
        background_color = driver.execute_script("return window.getComputedStyle(arguments[0], null).getPropertyValue('background-color');", element)
        
        #Bold and Red->Tier1
        if str(font_weight)=='700' and str(background_color)=='rgb(155, 49, 49)': 
            mystr=str(element.get_attribute("outerHTML")).strip()
            mytier=mystr[mystr.find('class='):mystr.find('>')]
            store_dic[mytier]='1'
      
        #Bold only->Tier2
        elif str(font_weight)=='700':
            mystr=str(element.get_attribute("outerHTML")).strip()
            mytier=mystr[mystr.find('class='):mystr.find('>')]
            store_dic[mytier]='2'

        #Tier3
        else:
            mystr=str(element.get_attribute("outerHTML")).strip()
            mytier=mystr[mystr.find('class='):mystr.find('>')]
            store_dic[mytier]='3'
  
  
    return store_dic

def GetValues(**kwargs):
    '''
    Using Beautiful Soup to Extract Desired Price Data from Trading Economics and Map it to a Tier Value.
    (Tier value obtained via GetTiers function)
    '''
    driver=kwargs.get('driver')
    store_dic=kwargs.get('store_dic')

    soup = BeautifulSoup(driver.page_source, 'html.parser') 
    table = soup.find('table', {'id': 'calendar'})
    # Check if the table is found and parse the content
    if table:
        table_data = []
        for row in table.find_all('tr'):
            foundtier='No'
            row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if True or row_data[1]=='Actual': #replace True with : (row_data[1] in get_country)
                row_prettify=str(row.prettify()) 
                if 'event-' in str(row_prettify):
                    try:
                        checkstr=str(row_prettify[row_prettify.find('<span'):row_prettify.find('</span>')])
                        foundtier=store_dic[checkstr[checkstr.find('class'):checkstr.find('>')]]
                    except Exception as e:
                        print(e)
                        foundtier='N/A'
                
                row_data.append(foundtier)
                table_data.append(row_data)
    else:
        print("Table with id 'calendar' not found.")
    return table_data

def AddTimezone(df,**kwargs):
    target_tz_value=kwargs.get('TargetTimezoneValue')
    target_tz_detail=all_timezones_details[target_tz_value]
    df[df.columns[0]]=pd.to_datetime(df[df.columns[0]])
    df[df.columns[0]] = df[df.columns[0]].dt.tz_localize(target_tz_detail)  # Timezone Aware
    #df[df.columns[0]] = df[df.columns[0]].dt.tz_convert(target_tz_detail)
    return df


def AddTimestamp(df,**kwargs):
    #print(df)
    df.columns = df.columns.str.strip().str.lower()
    df=df.dropna(how='all')
    df['date']=df['date'].ffill()
    df['only_date'] = df[df.columns[0]].apply(lambda x: " ".join(list(str(x).split())[1:]) if len(str(x))>8 else "" )
    df['only_date'] = df['only_date'].replace('',None)
    df['only_date'] = df['only_date'].ffill()

    df['only_time'] = df[df.columns[0]].apply(lambda x: list(str(pd.to_datetime(x)).split())[1] if len(str(x))==8 else "" )
    df['only_time'] = df['only_time'].replace('',None)
    df['only_time'] = df['only_time'].ffill()
    df['datetime']=df['only_date'].astype(str)+' '+df['only_time'].astype(str)
    df=df.dropna(subset=['only_time'])#drop row if time value is missing is None somewhere in the row.
    #print(df)
    df.datetime=pd.to_datetime(df.datetime)
    df['date']=df.datetime
    df.drop(inplace=True,axis=1,columns=['only_date','only_time','datetime'])
    df.rename({'date':'datetime'},inplace=True)
    #df.rename({'datetime',f'datetime as per {kwargs.get('TargetTimezoneName')}'})
    df.columns=df.columns.str.strip().str.upper()
    return AddTimezone(df,**kwargs)

def FilterByCountries(cal_df,**kwargs):
    """
    Filters dataframe/calendar to show data for only the countries 
    selected by the user.
    """
    # Get parameters from keyword arguments
    get_country=kwargs.get('get_country')

    # Return calendar with All Countries as it is
    if get_country==['All Countries']:
        Filtered_df=cal_df.copy()

    # Return calendar with filtered countries such that row's country value is in filtered country 
    # or 
    # row is the row with Date (not time)
    else:
        Filtered_df=cal_df.copy()
        for _,country_name in enumerate(Filtered_df.columns):
            if country_name in ['Country','country','COUNTRY']:
                break
        Filtered_df = cal_df[(cal_df[cal_df.columns[1]].isin(get_country))| (cal_df[country_name]=="")]

    return [cal_df,Filtered_df]


def StoreCalendar(cal_df,**kwargs):
        """
        Stores the calendar after creation and/or merging operations.
        Format: {Timezone}_{Filtered Countries}_trading_eco_cal_{start_date}_to_{end_date}.xlsx
        """
        
        # Get desired parameters for storing the merged calendar
        output_directory_name=kwargs.get('output_directory_name')
        os.makedirs(output_directory_name, exist_ok=True)
        output_file_name=None
        driver=kwargs.get('driver')
        MyTargetTimezoneName=kwargs.get('TargetTimezoneName')
        get_country=kwargs.get('get_country')
        display_country_boolean=kwargs.get('display_country')


        # Get new calendar with all countries' data and calendar with only selected countries data in a list alldf
        alldf= FilterByCountries(cal_df,**kwargs)

        # Merge the new calendar files with respective old calendar files sequentially
        for index,df in enumerate(alldf):
            # Dont display "Country" column if single country selected for filter.
            if display_country_boolean==False and index==1 and len(get_country)==1: 
                if 'COUNTRY' in df.columns:
                    df.drop('COUNTRY',axis=1,inplace=True)
                if 'Country' in df.columns:
                    df.drop('Country',axis=1,inplace=True)
            
            # Convert columns to CAPITAL
            df.columns = df.columns.str.upper()

            # Data Storage:
            # If Original df with all countries
            if index==0:
                output_file_name=MyTargetTimezoneName+'_'+'All Countries'+'_trad_eco_cal_'

            # If Filtered df with filterd countries
            elif index==1:
                output_file_name=kwargs.get('output_file_name')

            df=AddTimestamp(df,**kwargs)
                
            merged_name,merged_df=MergeCalendar(df,output_file_name,**kwargs)
            merged_df.columns=merged_df.columns.str.upper()
            #merged_df.rename(columns={'DATE':'DATETIME'},inplace=True)
            merged_df.reset_index(drop=True,inplace=True)
            
            merged_df[merged_df.columns[0]]=merged_df[merged_df.columns[0]].astype(str)
            merged_df.to_excel(os.path.join(output_directory_name,f"{merged_name}.xlsx"),index=False)
            merged_df.to_csv(os.path.join(output_directory_name,f"{merged_name}.csv"),index=False)

            # Ignore the below step. It was solving the error.
            merged_df2=pd.read_excel(os.path.join(output_directory_name,f"{merged_name}.xlsx"))
            merged_df2=merged_df2[~merged_df2['EVENTS'].isna()]
            merged_df2.to_excel(os.path.join(output_directory_name,f"{merged_name}.xlsx"),index=False)
            merged_df2.to_csv(os.path.join(output_directory_name,f"{merged_name}.csv"),index=False)
            
            
        driver.quit()
        return alldf


      
def MergeCalendar(new,new_path,**kwargs):
    #print(new)
    """
    Merges the newly obtained calendar data with old data (if there).
    If the timezone and filtered_countries both the parameters match, only then the merge happens. 
    Else, new files get created.
        
    """
    # Get output_directory name
    opdc=kwargs.get('output_directory_name')
    # Get timezone and country details from new dataframe
    new_details=(new_path.split('_'))[0:2]
    new_tz=new_details[0]
    new_country=new_details[1]

    # Scan Input_data for old calendar file path. If not found, take old calendar as empty dataframe
    old=pd.DataFrame()
    flag=False
    for entry in os.scandir(opdc):
        if entry.is_file() and entry.name.endswith('.xlsx') and (entry.name.split('_'))[0]==new_tz and (entry.name.split('_'))[1]==new_country:
            #print(entry)
            old_path=os.path.join(opdc,entry.name)
            old=pd.read_excel(old_path).copy()
            flag=True
            break
    
    merggeddf=new
    # If old calendar is not found, return new calendar as merged calendar
    if flag==True:
        try:
            for key in all_timezones:
                if all_timezones[key]==new_tz:
                    target_tz_value=key
                    target_tz_detail=all_timezones_details[target_tz_value]
                    break
            target_tz_detail=all_timezones_details[target_tz_value]
            old[old.columns[0]]=pd.to_datetime(old[old.columns[0]])
            old[old.columns[0]] = old[old.columns[0]].dt.tz_localize(target_tz_detail)  # Adjust as needed
            #old[old.columns[0]] = old[old.columns[0]].dt.tz_convert(target_tz_detail)

        except TypeError: #Already tz_aware.
            old[old.columns[0]] = old[old.columns[0]].dt.tz_convert(target_tz_detail)


        # Merge old and new calendar
        new.rename({'DATE':'DATETIME'})
        merggeddf=MergeCalendar_helper(old,new)

        # # Delete the old calendar file
        file_path=old_path #excel
        try:
            os.remove(file_path)
            print(f"{file_path} has been deleted successfully.")
        except FileNotFoundError:
            print(f"{file_path} does not exist.")
        except PermissionError:
            print(f"Permission denied: Cannot delete {file_path}.")
        except Exception as e:
            print(f"Error: {e}")
            trb.print_exc()
        # csv
        file_path2=old_path.replace('.xlsx','.csv') #excel_path only.
        try:
            os.remove(file_path2)
            print(f"{file_path2} has been deleted successfully.")
        except FileNotFoundError:
            print(f"{file_path2} does not exist.")
        except PermissionError:
            print(f"Permission denied: Cannot delete {file_path2}.")
        except Exception as e:
            print(f"Error: {e}")
            trb.print_exc()

    # Get start and end dates from merged calendar
    # finaldatesdf=(merggeddf[merggeddf[merggeddf.columns[0]].dt.date])
    # finaldatesdf.reset_index(drop=True,inplace=True)
    # Ensure the column is in datetime format
    merggeddf.rename({'DATE':'DATETIME'},inplace=True)
    print(merggeddf.columns)
    #merggeddf=merggeddf.sort_values(by=(merggeddf.columns[0]))
    merggeddf[merggeddf.columns[0]] = pd.to_datetime(merggeddf[merggeddf.columns[0]])
    merggeddf.index=merggeddf[merggeddf.columns[0]]
    merggeddf.sort_index(inplace=True)
    merggeddf.reset_index(drop=True,inplace=True)
    print('mergeddf',merggeddf)

    # Extract the first and last date (ignoring the time part)
    final_start_date = str(merggeddf[merggeddf.columns[0]].dt.date.iloc[0])
    final_end_date = str(merggeddf[merggeddf.columns[0]].dt.date.iloc[-1])

    finalname = new_tz+'_'+new_country+'_trad_eco_cal_'+final_start_date+'_to_'+final_end_date

    # Return the merged calendar
    return finalname,merggeddf
    

def MergeCalendar_helper(olddf,newdf):
    olddf.columns=olddf.columns.str.upper()
    newdf.columns=newdf.columns.str.upper()
    # Combine the old and new parts
    print(olddf.columns)
    print(newdf.columns)
    olddf.rename(columns={'DATE':'DATETIME'},inplace=True)
    newdf.rename(columns={'DATE':'DATETIME'},inplace=True)
    if 'COUNTRY' in olddf.columns:
        olddf=olddf[['DATETIME','COUNTRY','EVENTS','ACTUAL','PREVIOUS','CONSENSUS','FORECAST','TIER']]
        newdf=newdf[['DATETIME','COUNTRY','EVENTS','ACTUAL','PREVIOUS','CONSENSUS','FORECAST','TIER']]
    else:
        olddf=olddf[['DATETIME','EVENTS','ACTUAL','PREVIOUS','CONSENSUS','FORECAST','TIER']]
        newdf=newdf[['DATETIME','EVENTS','ACTUAL','PREVIOUS','CONSENSUS','FORECAST','TIER']]

    finalcsv = pd.concat([olddf,newdf], ignore_index=True,axis=0)
    print('final:')
    print(finalcsv)

    # Remove duplicate rows if new rows modify the previous ones.
    if 'country' or 'Country' or 'COUNTRY' in finalcsv.columns:
        finalcsv.drop_duplicates(subset=list(finalcsv.columns[:3]),keep='last',inplace=True)
    else:
        finalcsv.drop_duplicates(subset=list(finalcsv.columns[:2]),keep='last',inplace=True)
    #not_allowed=["",None]
    #finalcsv=finalcsv[(finalcsv.date is not in not_allowed) & (finalcsv.date is not in not_allowed)
    finalcsv.dropna(inplace=True,how='all') 
    finalcsv.index=finalcsv[finalcsv.columns[0]]
    finalcsv.index = pd.to_datetime(finalcsv.index)
    finalcsv.sort_index(inplace=True)
    finalcsv.reset_index(drop=True,inplace=True)
    finalcsv.rename(columns={'DATE':'DATETIME'},inplace=True)
    
    finalcsv.columns = finalcsv.columns.str.lower()
    # Check if all but 1 columns are none and remove those rows.
    
    return finalcsv


def FormatDate(DateStr):
    """
    Convert Date from default to YYYY-MM-DD if required. (Not used in the code)
    """
    months_dict={}
    month_name=['January', 'February', 'March', 'April', 
                'May', 'June', 'July', 'August', 'September',
                'October', 'November', 'December']
    month_val = ['0' + str(i) if i < 10 else str(i) for i in range(1, 13)]
    #month_val = [i[0:3] for i in month_name]
    for name,val in zip(month_name,month_val):
        months_dict[name]=val
    MyDateParts = DateStr.split(' ')
    return "-".join([MyDateParts[3], months_dict[MyDateParts[1]], MyDateParts[2]])



# All TimeZones:
all_timezones= {
    'UTC -12': 'BIT',            # Baker Island Time
    'UTC -11': 'SST',            # Samoa Standard Time
    'UTC -10': 'HST',            # Hawaii-Aleutian Standard Time
    'UTC -9': 'AKST',            # Alaska Standard Time
    'UTC -8': 'PST',             # Pacific Standard Time
    'UTC -7': 'MST',             # Mountain Standard Time
    'UTC -6': 'CST',             # Central Standard Time
    'UTC -5': 'EST',             # Eastern Standard Time
    'UTC -4': 'AST',             # Atlantic Standard Time
    'UTC -3': 'ART',             # Argentina Time
    'UTC -2': 'GST',             # South Georgia Time
    'UTC 0': 'GMT',              # Greenwich Mean Time
    'UTC +1': 'CET',             # Central European Time
    'UTC +2': 'EET',             # Eastern European Time
    'UTC +3': 'MSK',             # Moscow Standard Time
    'UTC +3:30': 'IRST',         # Iran Standard Time
    'UTC +5': 'PKT',             # Pakistan Standard Time
    'UTC +5:30': 'IST',          # Indian Standard Time
    'UTC +5:45': 'NPT',          # Nepal Time
    'UTC +6': 'BST',             # Bangladesh Standard Time
    'UTC +7': 'ICT',             # Indochina Time
    'UTC +8': 'CST',             # China Standard Time
    'UTC +9': 'JST',             # Japan Standard Time
    'UTC +9:30': 'ACST',         # Australian Central Standard Time
    'UTC +10': 'AEST',           # Australian Eastern Standard Time
    'UTC +10:30': 'LHST',        # Lord Howe Standard Time
    'UTC +12': 'FJT',            # Fiji Time
    'UTC +13': 'TOT',            # Tonga Time
    'UTC +14': 'LINT'            # Line Islands Time
}


all_timezones_details={
    'UTC -12': 'Etc/GMT+12',        # Baker Island Time
    'UTC -11': 'Pacific/Pago_Pago', # Samoa Standard Time
    'UTC -10': 'Pacific/Honolulu',  # Hawaii-Aleutian Standard Time
    'UTC -9': 'America/Anchorage',  # Alaska Standard Time
    'UTC -8': 'America/Los_Angeles', # Pacific Standard Time
    'UTC -7': 'America/Denver',     # Mountain Standard Time
    'UTC -6': 'America/Chicago',    # Central Standard Time
    'UTC -5': 'US/Eastern',   # Eastern Standard Time
    'UTC -4': 'America/Halifax',    # Atlantic Standard Time
    'UTC -3': 'America/Argentina/Buenos_Aires', # Argentina Time
    'UTC -2': 'Atlantic/South_Georgia', # South Georgia Time
    'UTC 0': 'UTC',                 # Coordinated Universal Time
    'UTC +1': 'Europe/Paris',       # Central European Time
    'UTC +2': 'Europe/Athens',      # Eastern European Time
    'UTC +3': 'Europe/Moscow',      # Moscow Standard Time
    'UTC +3:30': 'Asia/Tehran',     # Iran Standard Time
    'UTC +5': 'Asia/Karachi',       # Pakistan Standard Time
    'UTC +5:30': 'Asia/Kolkata',    # Indian Standard Time
    'UTC +5:45': 'Asia/Kathmandu',  # Nepal Time
    'UTC +6': 'Asia/Dhaka',         # Bangladesh Standard Time
    'UTC +7': 'Asia/Bangkok',       # Indochina Time
    'UTC +8': 'Asia/Shanghai',      # China Standard Time
    'UTC +9': 'Asia/Tokyo',         # Japan Standard Time
    'UTC +9:30': 'Australia/Darwin', # Australian Central Standard Time
    'UTC +10': 'Australia/Sydney',  # Australian Eastern Standard Time
    'UTC +10:30': 'Australia/Lord_Howe', # Lord Howe Standard Time
    'UTC +12': 'Pacific/Fiji',      # Fiji Time
    'UTC +13': 'Pacific/Tongatapu', # Tonga Time
    'UTC +14': 'Pacific/Kiritimati' # Line Islands Time

}