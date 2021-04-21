# Stage 1
## Updates
* Now it is possible to run the model with decent prf-values for longer texts.
* The only modifications were related to the train data which was recomposed using two auxiliar projects: "Keep Name and Type" (C++) and "GoodIndex" (Java) - both present in the "Data Sets" directory.
## Remarks
* Even though the performance seems to be increased significantly (based on the new prf-values) from Stage 0, it should be kept in mind the fact that the model still has flows.
## Problems
* As aforementioned, beginning with Stage 2, for some entities it will also be used the romanian model "ro_core_news_lg".
* Overlapping entities are not allowed. There are still some entities missing from the train data, problem that will be solved in Stage 2.
* At the moment, the solution would be to train the model independently for every type of entity because they won't overlap and it will be easier to use the "ro_core_news_lg" model for specific types.
## Content
* stage_1.ipynb - Python code that creates the model and tests it
* prf_1.xlsx - prf-values for Stage 1 model (it contains only one sheet because it was tested only on one data set to provide the most representative results)
