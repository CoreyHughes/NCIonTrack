# NCIonTrack
To make sure that every student is on track to graduate this project suggests that, National College of Ireland (NCI) in Dublin, Ireland should create a “dropout indicator” predictive model that is able to identify students who may be in need of extra attention and supports which can be then provided by the university. “NCI on Track” keeps track of grades, attendance, failures, health, and multiple other variables which are outlined further along in this report. 
The goal of this project includes a thorough validation and enhancement of the model, using new sources of student data and adding machine learning approaches to improve prediction. In this setting, I am attempting to predict students that are not likely to graduate on time and which ones are more urgently in need of attention. A student is labelled as “did not” graduate on time when:
- the student dropped out of university sometime after enrolling.
- the student is choosing to defer a semester.
The  main objective is to identify students who are at risk for not graduating on time. The model is trained on specific data from previous university classes and modelled their outcome from this. The hope is then to apply this predictive AI model on current students to predict their risk of falling behind. As a further step, survival analysis technique will be used to create a score associated with each of the students this will allow lectures or student support services to understand the urgency of a certain student. This will allow the university to identify who is most at risk of dropping out and using their resources where needed most. 
The dataset used to build this model contains automatic records of one set of (pretend) students tracked from the beginning of first year at university until the end. Each academic year in the dataset contains a range of features and can be put together into three categories: academic (e.g. gpas, and test scores), behavioural (e.g., time absent, number of suspensions,) and enrolment-related (e.g., new to school, new to the country). I also have flags that show each time a student falls behind and must repeat a subject.
The number one priority of this project is to: (i) gather a prediction and (ii) display the predictions through an interface. 
The variables used and labelled in this predictive model are as follows:
- school: student's school (NCI)
- 	sex: student's sex (binary: "F" - female or "M" - male)
- 	age: student's age (numeric: from 18 to 27)
- 	address: student's home address type (binary: "U" - urban or "R" - rural)
- 	famsize: family size (binary: "3" - less or equal to 3 or "4" - greater than 3)
- 	Pstatus: parent's cohabitation status (binary: "T" - living together or "A" - apart)
- 	Medu: mother's education (numeric: 0 - none, 1 - primary education 2 -“ junior certificate, 3 – leaving certificate or 4 - higher education)
- 	Fedu: father's education (numeric: 0 - none, 1 - primary education 2 -“ junior certificate, 3 – leaving certificate or 4 - higher education)
- 	Mjob: mother's job (nominal: "teacher", "health" “care related”, “civil services", "at_home" or "other")
- 	Fjob: father's job (nominal: "teacher", "health" “care related”, “civil services", "at_home" or "other")
- 	Reason: Reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
- 	Guardian: student's guardian (nominal: "mother", "father" or "other")
-	Traveltime: home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
-	Studytime: weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
-	Failures: number of past class failures (numeric)
-	Schoolsup: extra educational support (binary: yes or no)
-	Famsup: family educational support (binary: yes or no)
-	Paid: extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
-	Activities:  extra-curricular activities (binary: yes or no)
-	Nursery: attended nursery school (binary: yes or no)
-	Higher: wants to take higher education (binary: yes or no)
-	Internet: Internet access at home (binary: yes or no)
-	Romantic: with a romantic relationship (binary: yes or no)
-	Famrel: quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
-	Freetime: free time after school (numeric: from 1 - very low to 5 - very high)
-	Gout: going out with friends (numeric: from 1 - very low to 5 - very high)
-	Dalc: workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
-	Walc: weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
-	Health: current health status (numeric: from 1 - very good health to 5 - very bad health)
-	Absences: number of school absences (numeric: from 0 to 93)
-	Drop: did the student pass the final exam (binary: yes or no)
