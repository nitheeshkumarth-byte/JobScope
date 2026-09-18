r"""
resume_data.py — the canonical, ATS-friendly resume content.

This is the *source* data the resume generator (resume_generator.py) uses to
build a per-job .tex document. Keeping it as structured data (instead of
embedding in the generator) makes the resume easy to edit in one place.

WARNING: this file holds personal details (name, phone, email, links). It is
intended for your own, private use — do not commit it to a public repo.

The LaTeX template it drives is designed for Applicant Tracking Systems:
10pt article (2-column-safe layout avoided), standard \section*{...} headings,
one-column flow, no tables — ATS parsers read this cleanly.
"""

NAME = "Nitheesh Kumar Thadikamalla"
PHONE = "+91 7997457091"
EMAIL = "nitheeshkumar.th@gmail.com"
LOCATION = "Hyderabad, Telangana"

LINKS = {
    "linkedin": "https://www.linkedin.com/in/nitheesh-kumar-thadikamalla-a397b2235/",
    "github": "https://github.com/nitheeshkumarth-byte",
    "portfolio": "https://nitheeshkumarth.vercel.app/",
}

# Canonical objective; a job-tailored version is generated per listing.
OBJECTIVE_BASE = (
    "Computer Science graduate with hands-on backend development experience in "
    "Python, Django, Flask, and REST API design, backed by SQL/NoSQL databases "
    "(MySQL, MongoDB, SQLite) and AWS cloud deployment. Currently building "
    "AI-integrated backend systems as an AI/ML Engineer Intern, seeking a "
    "Software Developer/Engineer role to apply backend, cloud, and "
    "project-delivery skills."
)

# (Group name, skills string, [keywords that help ATS/job matching]).
# The generator reorders these groups so the most job-relevant appear first.
SKILLS = [
    ("Languages", "Python, PHP, JavaScript, SQL",
     ["python", "php", "javascript", "sql", "java"]),
    ("Backend / Web", "Django, Flask, FastAPI-ready REST API design, RESTful "
     "web services, Jinja2, SQLAlchemy, Flask-Migrate (Alembic), Gunicorn",
     ["django", "flask", "fastapi", "rest", "api", "jinja2", "sqlalchemy",
      "gunicorn"]),
    ("Databases", "MySQL, MongoDB, SQLite",
     ["mysql", "mongodb", "sqlite", "database", "db", "sql"]),
    ("Cloud / DevOps", "AWS (Bedrock, S3, OpenSearch, IAM, ECR, App Runner, "
     "Parameter Store), Terraform, Docker, Git, GitHub",
     ["aws", "s3", "opensearch", "iam", "ecr", "docker", "terraform", "git",
      "github"]),
    ("AI / ML", "RAG, LlamaIndex, Prompt Engineering, Google Gemini API, "
     "TensorFlow, Keras, OpenCV",
     ["llm", "rag", "llamaindex", "prompt", "gemini", "tensorflow", "keras",
      "opencv", "ai", "ml", "machine learning"]),
    ("Tools", "VS Code, Jupyter Notebook, Streamlit, Postman-style API "
     "testing, Unit Testing & Debugging",
     ["vscode", "jupyter", "streamlit", "postman", "testing", "debugging"]),
]

# (Role, dates, organisation, location, [bullets]).
EXPERIENCE = [
    ("AI/ML Engineer Intern", "Jul 2026 -- Present", "Quiddity Infotech LLC",
     "Hyderabad",
     ["Develop and integrate AI-driven backend features into production Python "
      "applications using REST APIs; collaborate on debugging, code reviews, "
      "and coding standards."]),
    ("Java Full Stack Developer (Training)", "May 2025 -- Nov 2025",
     "J-Spiders", "Hyderabad",
     ["Completed structured full-stack training in backend service design, "
      "database integration, and REST API development across Java and Python "
      "stacks."]),
    ("Web Development Intern", "Apr 2023 -- May 2023", "Corizo", "Bengaluru",
     ["Contributed to web application development tasks, gaining early "
      "exposure to backend logic and database queries."]),
]

# (Name, tech-stack meta line, [bullets]).
PROJECTS = [
    ("Recipe Box -- AI-Powered Recipe Manager",
     "Flask, SQLAlchemy, SQLite, Google Gemini API, Jinja2, JavaScript",
     ["Built a full-stack CRUD app (photo uploads, ingredient tracking, "
      "servings scaler) with RESTful routes, Flask-Migrate migrations, and "
      "Gemini API integration for AI features (dish-origin lookup, nutrition "
      "estimation, auto-fill), with result caching."]),
    ("RAG-Based Knowledge Assistant",
     "AWS Bedrock, OpenSearch, S3, LlamaIndex, Gradio, Docker, ECR, App Runner",
     ["Built an end-to-end Python pipeline -- S3 ingestion, "
      "chunking/embedding/indexing, query-time retrieval -- containerized "
      "with Docker, deployed to AWS App Runner via ECR with least-privilege "
      "IAM."]),
    ("ArticleGPT -- AI Blog Assistant",
     "Python, Google Gemini API, Streamlit, BeautifulSoup4",
     ["Built a web app that extracts article content from any blog URL, "
      "generates an AI summary, and answers follow-up questions via a "
      "Gemini-powered conversational interface with a Streamlit front end."]),
]

# (Title, dates, org with score, location).
EDUCATION = [
    ("B.Tech, Computer Science", "2020 -- 2024",
     "SRM University, AP -- CGPA/Score: 70%", ""),
]
# Extra education entry as a (label, years) pair, rendered like a project
# subheading by the generator (ATS-safe block, no tables).
EDUCATION_EXTRA = ("Intermediate (MPC), Narayana Junior College, Adibatla "
                   "-- Score: 90%", "2018 -- 2020")

CERTIFICATIONS = [
    "HackerRank -- Problem Solving; Introduction to Programming Using Java",
    "Introduction to Japanese Language and Culture",
    "Google Cloud -- Prompt Design in Vertex AI; Inspect Rich Documents with "
    "Gemini (Multimodal RAG Skill Badge)",
]

LANGUAGES = "Telugu (Native)  \\quad|\\quad  Hindi (Fluent)  \\quad|\\quad  English (Fluent)"


def flat_skill_keywords() -> list[str]:
    """Flatten every skill group's keyword list (used to auto-suggest skills
    and to match job descriptions even before a CV upload)."""
    out = []
    for _, _, kws in SKILLS:
        out.extend(kws)
    return out