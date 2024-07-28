from flask import Flask, render_template, request, jsonify, session
import os
import json
import pandas as pd
import base64
app = Flask(__name__, static_folder='static')
app.secret_key = "UAHjbd,sndnk;nfjdbfhbchdfdjcx,:l"
categories = {
    "others": {
        "display_name": "Others",
        "description": "Other miscellaneous categories.",
        "index": 0
    },
    "engineering": {
        "display_name": "Engineering",
        "description": "All aspects related to engineering.",
        "index": 1
    },
    "administration_hr": {
        "display_name": "Administration & Human Resources",
        "description": "Includes administration and human resources activities.",
        "index": 2
    },
    "data_science_ai": {
        "display_name": "Data Science & AI",
        "description": "Categories related to data science and artificial intelligence.",
        "index": 3
    },
    "business_management": {
        "display_name": "Business & Management",
        "description": "All business and management related activities.",
        "index": 4
    },
    "data_engineering": {
        "display_name": "Data Engineering",
        "description": "Includes data engineering and related fields.",
        "index": 5
    },
    "it_software_dev": {
        "display_name": "Information Technology (IT) & Software Development",
        "description": "Covers IT and software development areas.",
        "index": 6
    },
    "education_training": {
        "display_name": "Education & Training",
        "description": "Categories related to education and training.",
        "index": 7
    },
    "design_development": {
        "display_name": "Design & Development",
        "description": "Includes design and development related activities.",
        "index": 8
    }
}

experience_levels = {
    "mid_level": {
        "display_name": "Mid-level",
        "description": "Experience level for mid-level professionals.",
        "index": 0
    },
    "senior_level": {
        "display_name": "Senior-level",
        "description": "Experience level for senior-level professionals.",
        "index": 1
    },
    "executive": {
        "display_name": "Executive",
        "description": "Experience level for executives.",
        "index": 2
    },
    "entry_level": {
        "display_name": "Entry-level",
        "description": "Experience level for entry-level professionals.",
        "index": 3
    },
    "not_specified": {
        "display_name": "Not specified",
        "description": "Experience level not specified.",
        "index": 4
    }
}


def init_data(jobs_df, cvs_df, save_path="data/ground truth/ground truth1.csv", N_jobs=10):
    labeled_data = pd.DataFrame()
    categories = jobs_df["category"].unique()
    experience_levels = jobs_df["experience_level"].unique()
    return categories, experience_levels


def filter_cvs(): 
    cvs = {i:f"Hello, I am cv_{i}" for i in range(12)}
    return cvs

def job_reqs_extraction(job_description): 
    return "Minumun requirements: blablablabla, additional requirements : yaaref ya3ser 9ahwa"

def resume_formatter(cv): 
    return cv


JOBS_DF = ""
CVS_DF = ""
SAVE_PATH = ""
N_JOBS= 2

data = {
    'category': [3, 3, 3, 2],
    'experience_level': [0, 0, 0, 2],
    'Document': ['Document 1', 'Document 2', 'Document 3', 'Document 4'],
    'Index': [1, 2, 3, 4]
}

JOBS_DF = pd.DataFrame(data)

print(JOBS_DF)




@app.route('/')
def index():
    #categories,experience_levels = init_data(jobs_df = JOBS_DF, cvs_df = CVS_DF, save_path = SAVE_PATH, N_jobs=N_JOBS)
    return render_template('index.html',categories=categories,experiences=experience_levels)

@app.route('/visualize', methods=['POST'])
def visualize():
    selected_category = int(request.form['categorie'])
    selected_experience_level = int(request.form["experience"])
    print(selected_category, selected_experience_level)
    grouped_jobs = JOBS_DF[
        (JOBS_DF["category"] == selected_category) &
        (JOBS_DF["experience_level"] == selected_experience_level)
    ]

    selected_jobs = grouped_jobs.sample(n=min(N_JOBS, len(grouped_jobs)), random_state=42)
    idx = 0
    job = selected_jobs.iloc[idx]
    job_category_experience = f"Job Category: {selected_category}, Experience Level: {selected_experience_level}"
    job_description = job["Document"]
    index = job["Index"]
    job_content = f"{job_category_experience}\n\n{job_description} (Index: {idx})"
    job_file_path = "selected_job.txt"
    with open(job_file_path, "w") as job_file:
        job_file.write(job_content)


    session["index"] = idx
    session["current_category"] = selected_category
    session["selected_experience_level"] = selected_experience_level
    session["job_description"] = job_description
    return jsonify({"job_content" :job_content})

@app.route('/response', methods=['POST'])
def response():
    selected_response = request.form['response']
    if selected_response == "yes": 
        filtered_cvs_dict = filter_cvs()
        job_description = session.get("job_description")
        requirements = job_reqs_extraction(job_description)
        cvs = {}
        for index, cv in filtered_cvs_dict.items():
            formatted_cv = resume_formatter(cv)
            cvs[index] = formatted_cv
        return jsonify({"cvs": cvs, "requirements": requirements})

    else:
        current_index = session.get("index", 0)
        selected_category = session.get("current_category")
        selected_experience_level = session.get("selected_experience_level") 
        grouped_jobs = JOBS_DF[
        (JOBS_DF["category"] == selected_category) &
        (JOBS_DF["experience_level"] == selected_experience_level)
    ]
        selected_jobs = grouped_jobs.sample(n=min(N_JOBS, len(grouped_jobs)), random_state=42)
        current_index+=1
        session["index"] = current_index
        try:
            job = selected_jobs.iloc[current_index]
            job_category_experience = f"Job Category: {selected_category}, Experience Level: {selected_experience_level}"
            job_description = job["Document"]
            index = job["Index"]
            job_content = f"{job_category_experience}\n\n{job_description} (Index: {current_index})"
            job_file_path = "selected_job.txt"
            with open(job_file_path, "w") as job_file:
                job_file.write(job_content)
        except: 
            job_content = "No more sampled jobs fitting this category and number of required jobs. Change category/experience or increase n_jobs"
        
        return jsonify({"job_content" :job_content})
@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    kpi = request.form['kpi']
    comment = request.form['comment']
    
    # Save the comment and KPI to a text file
    with open('comments.txt', 'a') as file:
        file.write(f"KPI: {kpi}\nComment: {comment}\n\n")
    
    return "Comment submitted successfully!"


if __name__ == '__main__':
    app.run(debug = True)
