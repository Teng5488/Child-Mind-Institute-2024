# Child-Mind-Institute — Problematic Internet Use
偵測青少年網路使用成癮問題


### 1. 定義 X, Y

#### 🔹 解釋變數 (X)

####  Demographics
- `Basic_Demos-Enroll_Season`: Season of enrollment
- `Basic_Demos-Age`: Age of participant
- `Basic_Demos-Sex`: Sex of participant

####  Children's Global Assessment Scale
- `CGAS-Season`: Season of participation
- `CGAS-CGAS_Score`: Children's Global Assessment Scale Score

####  Physical Measures
- `Physical-Season`: Season of participation
- `Physical-BMI`: Body Mass Index (kg/m²)
- `Physical-Height`: Height (in)
- `Physical-Weight`: Weight (lbs)
- `Physical-Waist_Circumference`: Waist circumference (in)
- `Physical-Diastolic_BP`: Diastolic Blood Pressure (mmHg)
- `Physical-HeartRate`: Heart rate (beats/min)
- `Physical-Systolic_BP`: Systolic Blood Pressure (mmHg)

####  FitnessGram Vitals and Treadmill
- `Fitness_Endurance-Season`: Season of participation
- `Fitness_Endurance-Max_Stage`: Maximum stage reached
- `Fitness_Endurance-Time_Mins`: Time completed (minutes)
- `Fitness_Endurance-Time_Sec`: Time completed (seconds)

####  FitnessGram Child
- `FGC-Season`: Season of participation
- `FGC-FGC_CU(Zone)`: Curl up total (fitness zone)
- `FGC-FGC_GSND(Zone)`: Grip Strength (non-dominant)
- `FGC-FGC_GSD(Zone)`: Grip Strength (dominant)
- `FGC-FGC_PU(Zone)`: Push-up total (fitness zone)
- `FGC-FGC_SRL(Zone)`: Sit & Reach (left side)
- `FGC-FGC_SRR(Zone)`: Sit & Reach (right side)
- `FGC-FGC_TL(Zone)`: Trunk lift total (fitness zone)

####  Bio-electric Impedance Analysis
- `BIA-Season`: Season of participation
- `BIA-BIA_Activity_Level_num`: Activity Level
- `BIA-BIA_BMC`: Bone Mineral Content
- `BIA-BIA_BMI`: Body Mass Index
- `BIA-BIA_BMR`: Basal Metabolic Rate
- `BIA-BIA_DEE`: Daily Energy Expenditure
- `BIA-BIA_ECW`: Extracellular Water
- `BIA-BIA_FFM`: Fat Free Mass
- `BIA-BIA_FFMI`: Fat Free Mass Index
- `BIA-BIA_FMI`: Fat Mass Index
- `BIA-BIA_Fat`: Body Fat Percentage
- `BIA-BIA_Frame_num`: Body Frame
- `BIA-BIA_ICW`: Intracellular Water
- `BIA-BIA_LDM`: Lean Dry Mass
- `BIA-BIA_LST`: Lean Soft Tissue
- `BIA-BIA_SMM`: Skeletal Muscle Mass
- `BIA-BIA_TBW`: Total Body Water

####  Physical Activity Questionnaire (Adolescents)
- `PAQ_A-Season`: Season of participation
- `PAQ_A-PAQ_A_Total`: Activity Summary Score (Adolescents)

####  Physical Activity Questionnaire (Children)
- `PAQ_C-Season`: Season of participation
- `PAQ_C-PAQ_C_Total`: Activity Summary Score (Children)

####  反應變數 (Y)

####  PCIAT - Parent–Child Internet Addiction Test

- `PCIAT-Season`: Season of participation
- `PCIAT-PCIAT_01`: Child disobeys time limits for online use
- `PCIAT-PCIAT_02`: Neglects chores for online time
- `PCIAT-PCIAT_03`: Prefers online time over family time
- `PCIAT-PCIAT_04`: Forms new online relationships
- `PCIAT-PCIAT_05`: Complaints about child’s online time
- `PCIAT-PCIAT_06`: Grades suffer due to online use
- `PCIAT-PCIAT_07`: Checks email before doing other things
- `PCIAT-PCIAT_08`: Seems withdrawn since discovering internet
- `PCIAT-PCIAT_09`: Becomes secretive about online activities
- `PCIAT-PCIAT_10`: Sneaks online use
- `PCIAT-PCIAT_11`: Spends time alone online in room
- `PCIAT-PCIAT_12`: Receives strange phone calls from online friends
- `PCIAT-PCIAT_13`: Acts annoyed when disturbed online
- `PCIAT-PCIAT_14`: Appears more fatigued than before internet use
- `PCIAT-PCIAT_15`: Preoccupied with going back online
- `PCIAT-PCIAT_16`: Throws tantrums over online time limits
- `PCIAT-PCIAT_17`: Chooses online time over hobbies/outdoor interests
- `PCIAT-PCIAT_18`: Gets angry when online time is limited
- `PCIAT-PCIAT_19`: Chooses online time over friends
- `PCIAT-PCIAT_20`: Feels better online after being moody offline
- `PCIAT-PCIAT_Total`: Total Score

#### 將反應變數分成三種情況： PCIAT 01~20、PCIAT Total、Severe Injury Index(SII)







### 2.preprocessing 
- 針對 missing value 做處理，刪除或使用smote方法補足
- 刪除id或季節等等對於模型無太大幫助的變數
- 以KNN方法補足反應變數的缺失值，以達到數據完整性
- 刪除高相關性的變數，如保留BMI，刪除身高與體重等變數
  

### 3.模型/呈現方法
1. 模型
- XGBoost
- LightGBM 
- CatBoost
- MLP
- TabNet


2. 呈現方法
- 加權的 kappa score

### 4.分析結果
1. 以xgboost模型預測SII，keppa score分數達到0.353，為所有模型之最
2. 再考慮smote、k-fold等等方法，xgboost的keppa score能上升至0.393


