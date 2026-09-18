from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime

app = Flask(__name__)
app.secret_key = "my_secret_key_123"
app.permanent_session_lifetime = timedelta(hours=1)

# SQLite3 DB 파일명
DB_NAME = "users.db"


# DB 연결 함수
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# DB 및 users 테이블 생성
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# 기본 페이지는 로그인 페이지로 이동
@app.route("/")
def index():
    return redirect(url_for("login"))


# 회원가입
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("아이디와 비밀번호를 모두 입력하세요.")
            return redirect(url_for("register"))

        # 비밀번호 암호화
        hashed_password = generate_password_hash(password)

        # 현재 PC 시간 기준 가입일 저장
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
            "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
            (username, hashed_password, created_at)
            )

            conn.commit()
            conn.close()

            flash("회원가입이 완료되었습니다. 로그인해주세요.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("이미 존재하는 아이디입니다.")
            return redirect(url_for("register"))

    return render_template("register.html")


# 로그인
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

        user = cursor.fetchone()
        conn.close()

        # 암호화된 비밀번호 비교
        if user and check_password_hash(user["password"], password):
            session.permanent = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(url_for("home"))
        else:
            flash("아이디 또는 비밀번호가 올바르지 않습니다.")
            return redirect(url_for("login"))

    return render_template("login.html")


# 홈 페이지
@app.route("/home")
def home():
    if "username" not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for("login"))

    return render_template("home.html", username=session["username"])


# 회원가입 리스트
@app.route("/users")
def users():
    if "username" not in session:
        flash("로그인이 필요합니다.")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, created_at FROM users ORDER BY id DESC"
    )

    user_list = cursor.fetchall()
    conn.close()

    return render_template("users.html", users=user_list)


# 로그아웃
@app.route("/logout")
def logout():
    session.clear()
    flash("로그아웃되었습니다.")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()

    # 스마트폰 접속을 위해 host='0.0.0.0' 사용
    # PPT 예시에 맞춰 포트번호는 8080 사용
    app.run(host="0.0.0.0", port=8080, debug=True)