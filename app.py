from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and dataset
model = joblib.load('employee_turnover_model_shar.pkl')
data = pd.read_csv('simulated_employee_turnover_.csv')

# Maps for categorical values
gender_map = {'Male': 1, 'Female': 0}
education_order = {'High School': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4}
leadership_style = {'Supportive': 1, 'Authoritarian': 2, 'Democratic': 3}
org_support = {'Low': 2, 'Moderate': 1, 'High': 3}

# Full question texts
trait_questions = {
    "Conscientiousness": [
        "I am always prepared.", "I pay attention to details.", "I get chores done right away.",
        "I like order.", "I follow a schedule.", "I am exacting in my work.",
        "I make plans and stick to them.", "I complete tasks successfully.", "I am efficient."
    ],
    "Emotional_Resilience": [
        "I am able to adapt when changes occur.", "I have one close and secure relationship.",
        "Sometimes fate or God helps me.", "I can deal with whatever comes my way.",
        "Past successes give me confidence.", "I try to see the humorous side of things when I am faced with problems.",
        "Having to cope with stress can make me stronger.", "I tend to bounce back after illness, injury or other hardships.",
        "I believe most things happen for a reason.", "I make my best effort, no matter what.",
        "I believe I can achieve my goals, even if there are obstacles.", "Even when hopeless, I do not give up.",
        "In times of stress, I know where to find help.", "Under pressure, I stay focused and think clearly.",
        "I prefer to take the lead in problem-solving.", "I am not easily discouraged by failure.",
        "I think of myself as a strong person when dealing with life’s challenges and difficulties.",
        "I make unpopular or difficult decisions.", "I am able to handle unpleasant or painful feelings like sadness, fear, and anger.",
        "I have to act on a hunch.", "I have a strong sense of purpose in life.",
        "I feel like I am in control.", "I like challenges.", "I work to attain goals.",
        "I take pride in my achievements."
    ],
    "Stress_Levels": [
        "How often have you been upset because of something unexpected?",
        "How often have you felt unable to control the important things in your life?",
        "How often have you felt nervous and stressed?",
        "How often have you felt confident about your ability to handle personal problems?",
        "How often have you felt that things were going your way?",
        "How often have you found that you could not cope with all the things you had to do?",
        "How often have you been able to control irritations in your life?",
        "How often have you felt that you were on top of things?",
        "How often have you been angered because of things outside of your control?",
        "How often have you felt difficulties were piling up so high you could not overcome them?"
    ],
    "Job_Satisfaction": ["I find my work challenging and interesting.",
                         "I feel a sense of pride in doing my job.",
                         "I am satisfied with my current salary.",
                         "I believe I am fairly compensated for the work I do.",
                        "I have opportunities for advancement in this organization.",
                        "Promotions are handled fairly here.",
                        "My supervisor is competent in doing their job.",
                        "My supervisor treats me with respect.",
                        "I enjoy working with my co-workers.",
                        "My co-workers are helpful and friendly.",
                        "I am satisfied with the physical working conditions.",
                        "My work environment is comfortable and safe.", 
                        "There is good communication between management and employees.",
                        "I am informed about important decisions in the organization.",
                        "I receive adequate recognition for my work.",
                        "Good performance is acknowledged in my organization.",
                        "I feel secure about my job.",
                        "I rarely worry about losing my job.",
                        "I have enough freedom to decide how to do my work.",
                        "I am allowed to use my own judgment in doing my work."],
    "Productivity": [
        "I complete my tasks on time.", "I meet deadlines consistently.", "I manage my workload effectively.",
        "I deliver high-quality work.", "I maintain focus throughout the day.", "I exceed my job expectations.",
        "I effectively prioritize my work.", "I am productive even under pressure.", "I contribute significantly to team goals.",
        "I achieve my performance goals regularly.", "I adapt quickly to work-related changes.",
        "I strive for continuous improvement in my work."
    ],
    "Workload": [
        "The task required a high level of mental effort.", "The task was physically demanding.",
        "I felt rushed or under time pressure during the task.", "I performed the task successfully.",
        "I had to work hard to meet performance demands.", "I felt frustrated or annoyed while performing the task."
    ],
     "Emotion_Focused_Coping": [
        "I learn to live with it.",  # Acceptance
        "I accept that this has happened and that it can't be changed.",
        "I put my trust in God.",  # Religion
        "I try to find comfort in my religion or spiritual beliefs.",
        "I get upset and let my emotions out.",  # Venting Emotions
        "I express my negative feelings.",
        "I try to grow as a person as a result of the experience.",  # Positive Reinterpretation and Growth
        "I look for something good in what is happening.",
        "I give up trying to deal with it.",  # Behavioral Disengagement
        "I admit to myself that I can't deal with it and quit trying.",
        "I go to movies or watch TV to think about it less.",  # Mental Disengagement
        "I sleep more than usual to block it out."
    ],
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {}

        # Collect categorical data
        input_data['Gender'] = gender_map.get(request.form.get('Gender'))
        input_data['Education_Level'] = education_order.get(request.form.get('Education_Level'))
        input_data['Leadership_Style'] = leadership_style.get(request.form.get('Leadership_Style'))
        input_data['Organizational_Support'] = org_support.get(request.form.get('Organizational_Support'))

        # Collect numeric data
        numeric_columns = ['Age', 'Years_of_Experience', 'Compensation', 'Attendance']
        for col in numeric_columns:
            try:
                input_data[col] = float(request.form.get(col))
            except ValueError:
                return render_template('index.html', error=f"Invalid input for {col}")

        # Compute trait scores based on grouped Likert responses
        for trait, questions in trait_questions.items():
            # Get all the answers for the trait questions
            scores = []
            for i, question in enumerate(questions):
                try:
                    score = int(request.form.get(f"{trait}_{i}"))
                    scores.append(score)
                except ValueError:
                    return render_template('index.html', error=f"Invalid input for {trait} question {i + 1}")
            # Calculate the average score for the trait
            input_data[trait] = round(sum(scores) / len(scores), 2)

        # Reorder the input data to match the model's expected input
        correct_order = [
            'Age', 'Gender', 'Education_Level', 'Years_of_Experience', 'Workload',
            'Leadership_Style', 'Compensation', 'Organizational_Support',
            'Conscientiousness', 'Emotional_Resilience', 'Stress_Levels',
            'Emotion_Focused_Coping', 'Job_Satisfaction', 'Productivity',
            'Attendance'
        ]
        input_df = pd.DataFrame([input_data])[correct_order]

        # Model prediction
        prediction = model.predict(input_df)[0]
        label = "Attrition" if prediction == 1 else "No Attrition"
        input_df["Prediction"] = label

        # Save the prediction result
        try:
            pd.read_csv("prediction_results.csv")
            header = False
        except FileNotFoundError:
            header = True
        input_df.to_csv("prediction_results.csv", mode='a', index=False, header=header)

        return render_template('result.html', prediction=label)

    return render_template('index.html', trait_questions=trait_questions, data=data)

if __name__ == '__main__':
    app.run(debug=True)
