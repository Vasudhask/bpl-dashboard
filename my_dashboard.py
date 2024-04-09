import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import requests
from io import BytesIO

# Replace "github_username" with your GitHub username and "repo_name" with your repository name
url_sheet1 = 'https://raw.githubusercontent.com/Vasudhask/bpl-dashboard/main/yield_analysis_v0.1.xlsx'
url_sheet2 = 'https://raw.githubusercontent.com/Vasudhask/bpl-dashboard/main/defect_analysisutf.csv'

# Fetch data from the Excel file
response_sheet1 = requests.get(url_sheet1)
excel_content_sheet1 = BytesIO(response_sheet1.content)
df = pd.read_excel(excel_content_sheet1)

# Fetch data from the CSV file
response_sheet2 = requests.get(url_sheet2)
csv_content_sheet2 = BytesIO(response_sheet2.content)
df2 = pd.read_csv(csv_content_sheet2)


# Load the image
image = Image.open(r'C:\Users\Asus\Downloads\bpllogo.PNG')  # Replace 'path/to/your/image.jpg' with the actual file path
# Display the image
st.image(image, width=70)
st.markdown("<h1 style='font-size, 32px;color, grey;'>VERIFICATION AND VALIDATION: DATA ANALYSIS</h1>", unsafe_allow_html=True)


# app.py
#Define your products and types
products = {
    'Select a Product': [],
    'Vivid Vue': ['Vivid Vue 8', 'Vivid Vue 10', 'Vivid Vue 12'],
    'GenX': ['GenX3', 'GENX1'],
    'Relife': ['Relife900'],
    'EFLO': ['EFLO7', 'EFLO6'],
    'PRIME 320': ['PRIME 320'],
    'LF X-Ray': ['MRAD100', 'XRAD300', 'XRAD500'],
    'HF X-Ray': ['MRAD3.6+', 'MRAD3.5', 'MRAD5.0', 'HRAD32', 'HRAD50'],
    'C-Arm': ['CRAY PRO PLUS']
     }

categories = ['Patient Monitor','ECG','Defibrillator','AWS','X-Ray','C-Arm']
    
     
def function_defect():
    st.markdown("<h2 style='font-size, 16px;color, yellow;'>DEFECT ANALYSIS</h1>", unsafe_allow_html=True)

    #Define your products and types
    product_category = {
    'Select a Product Category': [ ],

    'Accessories': ['343PATDETTERAY', '12L3C002A'], 
    'Anaesthesia work station': [ 'MAMEFLO7', 'MAMPLN465', 'MAMEFLO7TAUTCLAA02SSPE', 'MAMVPRHLO', 'MAMEFLO6D', 'MAMEFLO7R', 'PENLON450MUAY', 'MAMEFLO6DMFM', 'MAMEFLO6-3GAS', 'MI5ENCIAMKSATPMRN', 'MAMEFLO6DR', 'MAMEFLO6PHMSCO2', 'MAMEFLO6', '5493GLIBTRUAY', 'PENLON450EUAY'], 
    'Assemblies and Sub-Assemblies': [ '382MXDIUAY', '382XMED1KMGUAY', 'RELIFE900R-SFG', '395XMED1KMGUAY'],
    'C-Arm': [ 'MBPLCRAYPXMEM1KMG', 'MBPLCRAYPROPLUSMXDI', 'MBPLCRAYP', 'MBPLCRAYP-MXDI', '382UAY', 'MBPLCRAYP-MXDI-W/OII', 'MBPLCRAYPROPLUSMXDI-W/OII', 'MBPLCRAYPROMXDI', 'MBPLCRAYPXMED1KMG-RA', 'MBPLCRAYPXMED1KMG', 'MBPLCRAYPROMXSI', 'MBPLCRAYP-RA-MXDI', 'MBPLCRAYPROPLUSMXSI', 'MBPLCRAYPROPLUSXMED1K'],
    'Cardiac Analyser': [ 'MHOLTER9800CH12', 'MHOLTER9800CH3', 'MABPMWT1SW', 'MHOLTERCH12NEOT1', 'MABPMWSW', 'MHOLTERCH12NEOT1SW'],
    'Colposcope': ['MCOLVIEWC1'],
    'Consumables others': ['357BATPACKAY', 'FGPSMPS2007', '13L1K039'],
    'Defibrillators- Mono phasic': [ 'MDF2509R', 'MDF2509'],
    'Defibrillators-Bi-Phasic': ['MRELIFE900R', 'MDF2617', 'MRL900AEDSPO2PACNIBPR', 'MRL900AEDSPO2R', 'MRELIFE900AEDR', 'MDF2628', 'MDF2617AEDR', 'MDF2617R', 'MRL900AEDSPO2NIBPR', 'MRELIFE900AEDPR', 'MR700AEDPR' ],
    'Diathermy': ['MCM2601', '423UAY', '425UAY', 'MSURGIXVS1', 'MSURGIXE3D', '428UAY', 'MSURGIXE2' ],
    'Electrocardiographs-Multi channel': ['M6208VIEWPLUS', 'MECG9108DTWFSW', 'MECGAR2100VIEW', 'MECG9108D', 'M8108VIEW', 'MECG200', '511UAY', 'M9108', 'MECGHD100+', 'MECG8108R', 'MECG6208VIEWC', 'MECGGENX3S', 'MECG6208VIEW', 'MECG7108', 'MECGGENX3', 'MECG9108DT' ],
    'Electrocardiographs-single channel': ['MECGGENX1', 'MECG6108TB', 'M108TDIGI' ],
    'Foetal Doppler': [ 'MFD9713', 'MFD9714'],
    'Foetal Monitor': ['MFM9854', '513UAY', 'MFM9855', '548UAY', 'MFM9852T', 'MFM9855TWINPROBE', 'MFM9856', 'MFM9853', 'MFM9852' ],
    'Home Care': ['91MED0412', '91MED284-B', '91MED555-M', '91MED197', '91MED700-M', '91MED753', '91MED271', '91MED183', '91MED300', '91MED698-M', '91MED300-MFG', '91MED735', '91MED501', '91MED549', '91MED419', '91MED168', '91MED129', '91MED171-M', '91MED242', '91MED698', '91MED700', '91MED184', '91MED590', '91MED776-M' ],
    'Lowenstein ventilators': ['BPLMVLSMEL300STR', 'BPLMVLSMEL600STR', 'MVLSMEL300UAY', 'MVLSMEL600', 'BPLMVLSMPRIVEN50C', 'BPLMVLSMEL600CTRW/OCOM', 'BPLMVLSMPRIVEN40FG', 'BPLMVLSMEL600CTR' ],
    'Miscellaneous': [ '400XRYGENPOSCOM'],
    'Mother Infant Care': ['696FLORETFBD1000', 'A61634BX', 'BPLFLORETFBT1000', 'BPLBLOSSOM100', 'BPLLEONIPHFOCOMPTR', 'BPLBLOSSOMBLANKET', 'MV0217001', 'BPLFLORETDTT1000', 'A62062', '696FLORET1000', 'BPLBLOSSOMDUO', 'BPLFLORETFBD1000', 'BPLFLORET100', '696FLORETDTT1000', 'BPLBLOSSOM10/100US', 'VHFO0217004', 'A61601', '698BLOSSOMDUOUAY', 'MVHFO0217004', '697BLOSSOM100UAY', '696FLORET100', 'BPLBLOSSOM10', '697BLOSSOM10/100USUAY', 'BPLFLORET1000', 'V0217001', 'BPLLEONIPCOMPTR' ],
    'New Consumable': ['91MED170' ],
    'OBSOLETE': ['ULTIMA5553PRO', 'PO5529S', 'MPMELIXO', 'ECG8108', 'MPMEXCELLOECOPLUS', 'BM5619', 'MPMEXCELLOPRIME', 'MPMACCURA5553', 'AEGIS5633', 'AGENTA5825', 'AWSPRIMA_SP2', 'ENDURA5815', 'MAXIMA5855', 'ECG9108C', 'MPMEXCELSIGNE12', 'FM9533/9534' ],
    'Oxygen concentrator': [ 'MOG4203SB', 'MOGOXY5NEOD', 'MOG4203B', 'MOGOXY10NEOD', 'MOGOXY5NEO', 'MOGOXY5NEOS', 'MOG4305'],
    'Patient monitors': ['MPM5588UPT', 'MPMAEGISPLUS', 'MPM5588UPMETCO2SS', 'ME12/17/100AGMW/OO2MDUPGKS', 'MPMCSC7', 'MPMVIVEDVUE12NSPO2IBPPHCO2SSPE', 'MPM5644SNIBPSPO2', 'MPMVIVEDVUE12NPHCO2MS', 'MPM5588UPDS', 'MEXCELOCO2SSADV', 'MBPLCNS8BED', 'MSS8NEO', 'MPMEV15TM', '699BLOSSOM10UAY', 'MPM5588UPCO2SSPE', 'MPMNSN7M', 'MPMVIVEDVUE8NSPO2', 'MPM5645CLAREO', 'MPME17CONSOL', 'MPMEV100T', 'MPO5531D', 'MPM5588UPDIBPCO2SSPE', 'MPMEV100TN', 'MPM5588UPDSPO2CO2SS', 'MPM5588IBPAGMPHNW/OO2SS', 'M5588UPDIBPCO2SS', 'MEXCELLODIBPCO2SS', 'MPM5588UPDTPHETCO2', 'MPME12CONSOLP', 'M5588IBP-2', 'MPM5588NCO2MPHSS', 'MPM5578ECOPRNTR', 'MPMEV15TNELIBPCO2SS', 'MPMEV15TNSP02CO2SSPE', 'MPMVIVIDVUE12MIBPAGMPHNW/OO2SS', 'MPMVIVEDVUE12NSPO2IBP', 'MCSC10NEO', 'MPMELE15', 'MPMSSS10T', 'MPMVIVEDVUE12MSPO2IBPPHCO2MS', 'MPMEXCELORADV', 'MBM5620', 'MPMCSC8', 'MPMSS10IBPCO2(SS)', 'MPMVIVEDVUE12TNSPO2IBPPHCO2SS', 'MPM5588UPDIBP', 'MPMEV15TMSP02IBPCO2SS(PE)', 'MCNSHWL', 'MPM5633B', 'MPM5588UPCO2SS', 'MCSC12NEO', 'MPMCSC12', 'MPMEV15TNELIBPCO2PE', 'MEXCELLOMSPO2', 'MPMVIVEDVUE12TMSPO2IBPPHCO2SS', 'MPMSS10IBP', 'MPM5579NTC02SS', 'MPMEV15TNSP02IBPAGMSSW/OO2', 'MPM5644O2', 'MPMSSS8', 'MPMEXCELLOD', 'MPMVIVEDVUE12TMSPO2', 'MPM5588UPMT', 'MPM5588TSPL', 'MPM5578ETCO2', 'MPME12NSPO2IBP', 'MBPLCNSPRO', 'MPMEV10DTIBP', 'MPMVIVEDVUE12NPHCO2SS', 'MPM5588UPTIBPCO2PHSS', 'MPMVIVEDVUE12MSPO2IBPPHCO2SS', 'MPMEV15TMSP02IBPAGMSSW/OO2', 'MPMSSS10', 'MPMCSC10', 'MPM5588MSP02', 'MPMVIVEDVUE10NSPO2', 'MPMVIVEDVUE8OXSPO2', 'MPMVIVEDVUE10MSPO2', 'MPMEV10TDVGA', 'MPM5644D', '506MUAY', 'MPMVIVEDVUE12MSPO2', 'MPMEV8', 'MPM5588UPDT', 'MPMEV8T', 'MPMEXCELLOADV', 'MPMEXCELLO', '696FLORETFBT1000', 'MPM5588UPDTCO2SS', '502DUAY', 'MPMVIVEDVUE12NSPO2IBPPHCO2SS', 'BPL-MASIMO-AGM LEMO-M', 'MPMVIVEDVUE12NSPO2', 'MPMSSS12IBPCO2MSENBLD', 'MPMVIVEDVUE12MSPO2IBP', 'MPM5588MSPO2IBP', '646UAY', 'MPM5588UIBPCO2SS', 'MPO5531N', 'ME12/17/100BISMDUPGKT', 'MPMVIVEDVUE10NSPO2IBP', 'MSS12NEO', 'MPMSSS12', 'MPM5588PRB', 'MSS10NEO', 'MPO5530', 'MPM5579NT', 'MPM5644', 'MPMVIVEDVUE12TNSPO2', 'MPM5578', 'MPM5588IBPAGMPHN', 'MPMCSC10W', 'MPMSSS10W', 'MCSC8NEO', 'MPM5588UPDIBPPE', 'MPMEV10TD', 'MPMSSS12IBP', 'MPMVIVEDVUE12OXSPO2', 'MPMVIVEDVUE12MSPO2PHCO2SS', 'MPMNSN7N', 'BPL-MASIMO-AGM O2 LEMO-M', 'MPMVIVEDVUE8MSPO2', 'MPME12CONSOL' ],
    'Penlon-Anaesthesia work station': ['MAMPLN450E', 'PENLON:PRIMASP102', 'MAMPLN450M', 'P320DGAFMS', 'MAMPLNPRI320A', 'MAMPLN451MRI', 'PENLON451M15UAY', 'FG5008230', 'MAMPLNPRI320AAGMPE', 'PENLON451MUAY', 'MAMPLN451MRIMI5', 'MAMPLN465AGMO2SSAUTOID', 'MAMPLNPRI320AAGMW/OO2MS', 'MAMPLN465W/OAGM', 'MAMPLN465AGMW/OO2SSAUTOID' ],
    'SPARES': ['363PWRPCBASY', '382SCBPCBAY', 'BPL-PENLON-5008190', '371PADAY' ],
    'STS-Stress test system': [ 'MSTS-NEO', 'MDYNASTS', 'MACQUNITNEO-BT', 'MSTS-LITE', 'MACQUNITNEO', 'MACQUNIT', 'MSTSTMILLU-BPL'],
    'Syringe pump': ['SYRPUMP4502', 'SYRPUMP4503', 'BPLSYRPUMP', 'BPLSYRPUMP1', 'BPLWKHP80 MRI' ],
    'UNIT ACCESSORIES': [ 'SAM1907100'],
    'UltraSound': [ 'MECUBE-12', 'M10001301', 'MECUBE-X70', 'M10001302', 'MECUBE-8DCW', 'MECUBE-7', 'MECUBE-8CW-LE', 'MECUBE-15 PW', 'ECUBE8LEUAY', 'M10002003', 'MECUBE-8CW', 'MECUBE-5', 'MECUBE-15', 'MECUBE-X90', 'M10002004', 'MECUBE-I7CW', 'MECUBE-5WN', 'MECUBE-8', 'MUSSCANPLUS', 'MECUBE-11(D)', 'MECUBEI7', 'MECUBE-9DIA', 'ECUBE5UAY', 'MECUBE-8LE', 'MECUBE-8DIA'],
    'Volumetric Pump': [ '527UAY', 'VOLPUMP4504', 'VOLPUMP4501', 'BPLVOLPUMP1'],
    'X-ray': [ 'MBPLMRAD3.6+DR-APR', 'MBPLMRAD5.0', 'MBPLHRAD50GEN-DR2KIT', 'MBPLXRAD500-DR1KIT', 'MBPLXRAD100FC', 'MBPLHRAD32-DR1', 'MBPLEVS4343-WP-ASM', 'MBPLHRAD40-FC', 'MBPLEXPD4343-PASM', 'MBPLHRAD32GEN-DR1PRIME', 'MBPLHRAD32GEN-DR1KIT', 'MBPLMRAD3.6DR-APR', 'MBPLMRAD3.6+', 'CASTRYFLOTOTAB', 'MBPLPLANO+GRID8:1-BPL', 'MBPLHRAD32PXR-FC', 'MBPLEVS3643-D', 'MBPLXRAD300FC', 'MBPLMULTIO-NT', 'MBPLH-RAD50', 'MBPLXRAD300-DR1KIT', 'MBPLEVS3643-WP-ASM', 'MBPLXRAD500FC', 'MBPLMRAD100', 'MBPLEVS3643-WP', 'MBPLEVS3643', 'MBPLHRAD32PXR-FCDR', 'MBPLPLANO+GRID6:1-BPL', 'MBPLMRAD3.5DR', 'MBPLMRAD5.0DR', 'MBPLMRAD3.6', 'MBPLHRAD40-DR1', 'MBPLXRAD300-DR1'],
     }
    
    selected_product = st.selectbox('Select Product Category', list(product_category.keys()))
    if st.checkbox('NEXT'):
        selected_type = st.selectbox(f'Select configuration for {selected_product}', product_category[selected_product])
        st.write(f'Selected Product is {selected_product} and Selected Type is {selected_type}')

        if st.button('CONTINUE'):
            try:
                #df2 = pd.read_csv(r"C:\Users\Asus\Downloads\defect_analysisutf.csv")
                filtered_df2 = df2[df2['Product (Ticket ID)']==selected_type]
                result = filtered_df2.groupby('Work Description (Ticket)')['Month'].value_counts().unstack(fill_value=0)
                result['count'] = result.sum(axis=1)
                result
                st.write("Please note that the missing months indicate 'No documented defect in the configuration during the whole month'")

            except UnicodeDecodeError:
                pass


def function_performance():
    st.markdown("<h2 style='font-size, 16px;color, blue;'>PERFORMANCE ANALYSIS</h1>", unsafe_allow_html=True)
    # Read the excel file. Add new columns.
    #df=pd.read_excel(r"C:/Users/Asus/Downloads/yield_analysis_v0.1.xlsx", engine='openpyxl')
    df=df.fillna(' ')
    df['yield_pc'] = ''
    df['dpmo'] = ''
    df['sigma_level'] = ''

    def calculate_yield(defects,units,opportunities):
         yield_pc = (1-(defects/(units*opportunities)))*100
         dpmo = (defects/(units*opportunities))*1000000
         if dpmo == 0:
              dpmo = 4         
         sigma_level=norm.ppf(1-(dpmo)/1000000)+1.5
         return yield_pc , dpmo, sigma_level

    for i in range(0, df.shape[0]):
         df['yield_pc'][i] = calculate_yield(df['defects'][i],df['units'][i],df['opportunities'][i])[0]
         df['dpmo'][i] = calculate_yield(df['defects'][i],df['units'][i],df['opportunities'][i])[1]
         df['sigma_level'][i]  = calculate_yield(df['defects'][i],df['units'][i],df['opportunities'][i])[2]

    def configuration_viz(configuration_name):
         filtered_df = df[df['configuration'] == configuration_name] # Filter the DataFrame to select rows where 'Model Combination' matches the user input
         return filtered_df
    
    def type_viz(model_type):
         filtered_type_df = df[df['type'] == model_type]
         filtered_config_df = []
         spec_conf_names = set(filtered_type_df['configuration'])
         for spec_conf_name in spec_conf_names:
              filtered_config_df.append(configuration_viz(spec_conf_name))
         return filtered_config_df

    def newfunction (model_type):
         filtered_type_df = df[df['type'] == model_type]
         spec_conf_names = set(filtered_type_df['configuration'])
         return spec_conf_names

    # Convert the 'months' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%b-%y')

    def moving_average_all(df, n):
         result = {}
         categories = df['category'].unique()
         for cat in categories:
              cat_data = df[df['category'] == cat]
              last_n_months = cat_data['date'].nlargest(n)
              avg_values = {}
              for col in ['yield_pc', 'dpmo', 'sigma_level']:
                   sum_last_n_values = cat_data[cat_data['date'].isin(last_n_months)][col].sum()
                   avg_values[col] = sum_last_n_values / n
              result[cat] = avg_values
              new_df_result = pd.DataFrame.from_dict(result, orient='index')
              new_df_result.columns = ['yield_pc', 'dpmo', 'sigma_level']
         return new_df_result


    # Convert the 'months' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%b-%y')

    def moving_average_specific(df, selected_category, n):
         result = {}
         cat_data = df[df['category'] == selected_category]
         last_n_months = cat_data['date'].nlargest(n)
         avg_values = {}
         for col in ['yield_pc', 'dpmo', 'sigma_level']:
              sum_last_n_values = cat_data[cat_data['date'].isin(last_n_months)][col].sum()
              avg_values[col] = sum_last_n_values / n
         result[selected_category] = avg_values
         df_result = pd.DataFrame.from_dict(result, orient='index')
         df_result.columns = ['yield_pc', 'dpmo', 'sigma_level']
         return df_result

    # Streamlit app
    #def main():
     
    outer_select = st.selectbox('Select Desired Performance Analysis',['Product-Wise','Category-Wise'])
    if st.checkbox('NEXT'):
          if outer_select == 'Product-Wise':
                selected_product = st.selectbox('Select a Product', list(products.keys()))
                if st.checkbox('CONTINUE'):
                     selected_type = st.selectbox(f'Select Model for {selected_product}', products[selected_product])
                     st.write(f"The Configurations present in {selected_type} are {newfunction(selected_type)}")
                     if st.button('View Graphs',key='first'):
                          filtered_config_list = type_viz(selected_type)
                          for i in range(0, len(filtered_config_list)):
                               filtered_config_df = pd.DataFrame(filtered_config_list[i])
                
                          #Plotting the Yield% bar graph
                          fig, ax = plt.subplots(figsize=(8, 4))
                          bars = ax.bar(filtered_config_df['date'], filtered_config_df['yield_pc'], width=9) 
                          plt.xticks(rotation=90,fontsize=8)
                          plt.yticks(fontsize=8)
                          plt.gca().set_ylim(0, 100)
                          plt.xlabel('Months',fontsize=8)
                          plt.ylabel('Yield%',fontsize=8)
                          plt.title(f'Yield% Across Months',fontsize=8,pad=20)
                                            
                          for bar in bars:
                                    height = bar.get_height()
                                    ax.annotate('{:.2f}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=6)

                          ax.grid(True) 
                          st.pyplot(fig)
                          st.write("The month with the highest Yield% is:",df[df['yield_pc'] == df['yield_pc'].max()]['date'].head(1))
                          st .write("The month with lowest Yield% is:",df[df['yield_pc'] == df['yield_pc'].min()]['date'].head(1))
                                
                        
                          #DPMO 
                          fig2, ax2 = plt.subplots(figsize=(8, 4))
                          newbars = ax2.bar(filtered_config_df['date'], filtered_config_df['dpmo'] ,width=9)
                          plt.xticks(rotation=90,fontsize=8)
                          plt.yticks(fontsize=8)
                          plt.gca().set_ylim(0, max(filtered_config_df['dpmo']))
                          plt.xlabel('Months',fontsize=8)
                          plt.ylabel('DPMO',fontsize=8)
                          plt.title(f'DPMO Across Months',fontsize=8,pad=20)

                          for newbar in newbars:
                                height_2 = newbar.get_height()
                                ax2.annotate('{:.2f}'.format(height_2), xy=(newbar.get_x() + newbar.get_width() / 2, height_2),xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=6)
                            
                          ax2.grid(True)    
                          st.pyplot(fig2)
                          st.write("The month with the highest DPMO is:",df[df['dpmo'] == df['dpmo'].max()]['date'].head(1))
                          st .write("The month with lowest DPMO is:",df[df['dpmo'] == df['dpmo'].min()]['date'].head(1))
                            

                          #Sigma Level Plot
                          fig3, ax3 = plt.subplots(figsize=(8, 4))
                          colors = ['red' if value < 3.5 else 'blue' for value in filtered_config_df['sigma_level']]
                          bars_3 = ax3.bar(filtered_config_df['date'], filtered_config_df['sigma_level'],width=9,color=colors) 
                          plt.xticks(rotation=90,fontsize=8)
                          plt.yticks(fontsize=8)
                          plt.gca().set_ylim(0, max(filtered_config_df['sigma_level']))
                          plt.xlabel('Months',fontsize=8)
                          plt.ylabel('Sigma Level',fontsize=8)
                          plt.title(f'Sigma Level Across Months',fontsize=8,pad=20)
                          for bar_3 in bars_3:
                                height_3 = bar_3.get_height()
                                ax3.annotate('{:.2f}'.format(height_3), xy=(bar_3.get_x() + bar_3.get_width() / 2, height_3),xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=6)
                    
                          ax3.grid(True)
                          st.pyplot(fig3)
                          st.write('Please note: Sigma Level below 3.5 value is marked red.')
                          st.write("The month with the highest Sigma Level is:",df[df['sigma_level'] == df['sigma_level'].max()]['date'].head(1))
                          st.write("The month with lowest Sigma Level is:",df[df['sigma_level'] == df['sigma_level'].min()]['date'].head(1))

                                #THE FIRST TYPE WISE STEP GETS OVER

          if outer_select == 'Category-Wise':
                recent_category_select = st.selectbox('Select Desired Analysis',['All Categories','Specific Category'])
                if st.checkbox('PROCEED'):
                      if recent_category_select == 'Specific Category':
                            selected_category = st.selectbox('Select a Category', categories)
                            n = st.slider("Select desired n-month analysis", min_value=1, max_value=12, value=1)

                            #n_str = st.text_input("Enter desired n-month analysis")
                            #n = int(n_str) if n_str.isdigit() else None
                            if st.checkbox('OK, NEXT'):
                                 st.write(f"The {n}-month analysis for {selected_category} is: ")
                                 st.write(moving_average_specific(df, selected_category, n))
                            # st.write(yield_analysis.category_wise(selected_category))
                            # st.write("Please note: The 'avg' indicates the average value across all months")

                      if recent_category_select == 'All Categories':
                            # n_str = st.text_input("Enter desired n-month analysis")
                            # n = int(n_str) if n_str.isdigit() else None
                            n = st.slider("Select desired n-month analysis", min_value=1, max_value=12, value=1)
                            new_df_result = moving_average_all(df, n)
                            st.write(new_df_result)

                            if st.checkbox('GRAPHS'):
                              
                              # Plotting yield%
                              figg, axx = plt.subplots(figsize=(8, 4))
                              barss = axx.bar(new_df_result.index, new_df_result['yield_pc'], color='blue') 
                              plt.xticks(rotation=45,fontsize=8)
                              plt.yticks(fontsize=8)
                              plt.gca().set_ylim(0, 100)
                              plt.xlabel('Category')
                              plt.ylabel('Yield% Values')
                              plt.title('Yield% Values for different categories')
                                            
                              for bar in barss:
                                    height = bar.get_height()
                                    axx.annotate('{:.2f}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=6)

                              axx.grid(True) 
                              st.pyplot(figg)
                              
                
                              # Plotting dpmo
                              figg2, axx2 = plt.subplots(figsize=(8, 4))
                              newbarss = axx2.bar(new_df_result.index, new_df_result['dpmo'], color='green')
                              plt.xticks(rotation=45,fontsize=8)
                              plt.yticks(fontsize=8)
                              #plt.gca().set_ylim(0, max(filtered_config_df['dpmo']))
                              plt.xlabel('Category')
                              plt.ylabel('DPMO Values')
                              plt.title('DPMO Values for different categories')
                              plt.xticks(rotation=45)
                            

                              for newbar in newbarss:
                                height_2 = newbar.get_height()
                                axx2.annotate('{:.2f}'.format(height_2), xy=(newbar.get_x() + newbar.get_width() / 2, height_2),xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=6)
                            
                              axx2.grid(True)    
                              st.pyplot(figg2)


                              # Plotting sigma level
                              figg3, axx3 = plt.subplots(figsize=(8, 4))
                              colors = ['red' if value < 3.5 else 'blue' for value in new_df_result['sigma_level']]
                              barss_3 = axx3.bar(new_df_result.index, new_df_result['sigma_level'], color=colors) 
                              plt.xlabel('Category')
                              plt.ylabel('Sigma Level Values')
                              plt.title('Sigma Level Values for different categories')
                              plt.xticks(rotation=45)

                            #   for i, value in enumerate(new_df_result['sigma_level']):
                            #         plt.text(i, value, str(round(value, 2)), ha='center', va='bottom')
                              for bar_3 in barss_3:
                                height_3 = bar_3.get_height()
                                axx3.annotate('{:.2f}'.format(height_3), xy=(bar_3.get_x() + bar_3.get_width() / 2, height_3),xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',fontsize=6)
                    
                              axx3.grid(True)
                              st.pyplot(figg3)
                              st.write('Please note: Sigma Level below 3.5 value is marked red.')



def main():
 select_analysis = st.selectbox('Select Analysis', ['Performance Analysis','Defect Analysis'])
 
 if st.checkbox('Start Analysis'):
      if select_analysis == 'Defect Analysis':
           function_defect()

      if select_analysis == 'Performance Analysis':
           function_performance()

if __name__ == '__main__':
    main()