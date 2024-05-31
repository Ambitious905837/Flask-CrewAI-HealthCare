from flask import Flask, request, render_template, jsonify
from crewai import Crew
from tasks import Tasks
from agents import Agents

app = Flask(__name__, static_folder='static')

tasks = Tasks()
agents = Agents()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    analyze_task = data['analyze_task']
    summary_task = data['summary_task']

    researcher_agent = agents.researcher_agent()
    writer_agent = agents.writer_agent()

    analyze_healthcare_ai_news = tasks.anaylze_healthcare_ai_news(researcher_agent, analyze_task)
    write_summaries = tasks.write_summaries(writer_agent, summary_task)

    crew = Crew(
        agents=[researcher_agent, writer_agent],
        tasks=[
            analyze_healthcare_ai_news,
            write_summaries,
        ]
    )

    result = crew.kickoff()

    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
