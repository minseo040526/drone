"""
드론-트럭 VRPTW 최적화 웹앱
============================
Streamlit 단일 파일 앱.
실행: streamlit run app.py
배포: Streamlit Community Cloud (https://share.streamlit.io)
"""

import streamlit as st
import math
import random
import time
import sqlite3
import io
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 그래프는 영어 라벨 사용 — 폰트 의존성 없음
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="드론-트럭 VRPTW 최적화",
    page_icon="🚁",
    layout="wide",
)

# ============================================================
# 간단 로그인 (데모용)
# ============================================================
VALID_USERS = {
    "demo": "demo1234",
    "admin": "admin1234",
}

def login():
    """로그인 페이지. 로그인 성공 전까지 메인 앱 숨김."""
    if st.session_state.get("logged_in"):
        return True
    
    st.title("🚁 드론-트럭 VRPTW 최적화 시스템")
    st.caption("로그인이 필요합니다")
    
    with st.form("login_form"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("아이디", placeholder="demo")
        with col2:
            password = st.text_input("비밀번호", type="password", placeholder="demo1234")
        submit = st.form_submit_button("로그인", type="primary", use_container_width=True)
        
        if submit:
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
    
    with st.expander("💡 테스트 계정"):
        st.code("아이디: demo / 비밀번호: demo1234\n아이디: admin / 비밀번호: admin1234")
    
    return False


# 로그인 체크 — 이 아래 모든 코드는 로그인 후에만 실행됨
if not login():
    st.stop()

# ============================================================
# 데이터 구조 (노트북에서 복사)
# ============================================================
@dataclass
class Customer:
    id: int; x: float; y: float
    demand: float; ready_time: float; due_time: float; service_time: float

@dataclass
class VehicleSpec:
    name: str; capacity: float; speed: float
    max_distance: float; cost_per_distance: float; fixed_cost: float

@dataclass
class Solution:
    truck_routes: List[List[int]] = field(default_factory=list)
    drone_routes: List[List[int]] = field(default_factory=list)
    def all_customers(self):
        result = []
        for r in self.truck_routes + self.drone_routes: result.extend(r)
        return result
    def copy(self):
        return Solution(
            truck_routes=[r[:] for r in self.truck_routes],
            drone_routes=[r[:] for r in self.drone_routes])


class VRPTWProblem:
    def __init__(self, customers, truck_spec, drone_spec,
                 num_trucks=12, num_drones=8):
        self.customers = customers
        self.depot = customers[0]
        self.n_customers = len(customers) - 1
        self.truck_spec = truck_spec
        self.drone_spec = drone_spec
        self.num_trucks = num_trucks
        self.num_drones = num_drones
        self.tw_penalty = 10.0
        self.cap_penalty = 5000.0
        self.dist_penalty = 5000.0
        n = len(customers)
        self.dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.dist[i][j] = math.hypot(
                        customers[i].x - customers[j].x,
                        customers[i].y - customers[j].y)

    @classmethod
    def from_solomon_text(cls, text, num_trucks=12, num_drones=8,
                           drone_cap_ratio=0.15, drone_speed=2.0,
                           drone_max_dist=50.0):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        truck_capacity = 200.0
        for line in lines[:10]:
            parts = line.split()
            if len(parts) == 2 and all(p.replace(".", "").isdigit() for p in parts):
                truck_capacity = float(parts[1]); break
        customers = []
        for line in lines:
            parts = line.split()
            if len(parts) == 7:
                try:
                    nums = [float(p) for p in parts]
                    customers.append(Customer(
                        id=int(nums[0]), x=nums[1], y=nums[2],
                        demand=nums[3], ready_time=nums[4],
                        due_time=nums[5], service_time=nums[6]))
                except ValueError: continue
        truck = VehicleSpec("Truck", truck_capacity, 1.0, float("inf"), 1.0, 50.0)
        drone = VehicleSpec("Drone", truck_capacity * drone_cap_ratio,
                            drone_speed, drone_max_dist, 0.5, 20.0)
        return cls(customers, truck, drone, num_trucks, num_drones)

    def _evaluate_route(self, route, spec):
        if not route: return 0.0, 0.0, 0.0, 0.0
        full = [0] + route + [0]
        total_dist = 0.0
        load = sum(self.customers[c].demand for c in route)
        time_ = 0.0; tw_violation = 0.0
        for i in range(len(full) - 1):
            a, b = full[i], full[i+1]
            d = self.dist[a][b]
            total_dist += d
            time_ += d / spec.speed
            if b != 0:
                cust = self.customers[b]
                if time_ < cust.ready_time: time_ = cust.ready_time
                if time_ > cust.due_time: tw_violation += (time_ - cust.due_time)
                time_ += cust.service_time
        return (total_dist, tw_violation,
                max(0.0, load - spec.capacity),
                max(0.0, total_dist - spec.max_distance))

    def evaluate(self, solution, return_breakdown=False):
        total_cost = total_dist_t = total_dist_d = 0.0
        total_tw = total_cap = total_batt = 0.0
        n_t = n_d = 0
        for route in solution.truck_routes:
            if not route: continue
            n_t += 1
            d, tw, cap, _ = self._evaluate_route(route, self.truck_spec)
            total_dist_t += d
            total_cost += d * self.truck_spec.cost_per_distance + self.truck_spec.fixed_cost
            total_tw += tw; total_cap += cap
        for route in solution.drone_routes:
            if not route: continue
            n_d += 1
            d, tw, cap, batt = self._evaluate_route(route, self.drone_spec)
            total_dist_d += d
            total_cost += d * self.drone_spec.cost_per_distance + self.drone_spec.fixed_cost
            total_tw += tw; total_cap += cap; total_batt += batt
        visited = set(solution.all_customers())
        unvisited = set(c.id for c in self.customers[1:]) - visited
        unvisited_penalty = len(unvisited) * 10000.0
        penalty = (total_tw * self.tw_penalty + total_cap * self.cap_penalty
                   + total_batt * self.dist_penalty + unvisited_penalty)
        fitness = total_cost + penalty
        if return_breakdown:
            return dict(fitness=fitness, cost=total_cost, penalty=penalty,
                        truck_distance=total_dist_t, drone_distance=total_dist_d,
                        n_trucks=n_t, n_drones=n_d, tw_violation=total_tw,
                        cap_violation=total_cap, batt_violation=total_batt,
                        unvisited=len(unvisited))
        return fitness


# ============================================================
# 초기해 + 이웃 연산자 (간략화 버전)
# ============================================================
def random_solution(problem, seed=None):
    if seed is not None: random.seed(seed)
    ids = [c.id for c in problem.customers[1:]]
    random.shuffle(ids)
    dcap = problem.drone_spec.capacity
    tr = [[] for _ in range(problem.num_trucks)]
    dr = [[] for _ in range(problem.num_drones)]
    for cid in ids:
        # 드론이 있고 + 용량 적합 + 30% 확률
        if (problem.num_drones > 0 and dr and
            problem.customers[cid].demand <= dcap and random.random() < 0.3):
            dr[random.randint(0, problem.num_drones - 1)].append(cid)
        elif tr:
            tr[random.randint(0, problem.num_trucks - 1)].append(cid)
        else:
            # 트럭도 드론도 없음 — 미방문 (페널티로 처리됨)
            pass
    return Solution(truck_routes=tr, drone_routes=dr)

def greedy_solution(problem):
    dcap = problem.drone_spec.capacity
    # 드론이 없는 baseline 시나리오면 모든 고객을 트럭에 배정
    if problem.num_drones == 0:
        dcand = []
        tcand = [c.id for c in problem.customers[1:]]
    else:
        dcand = [c.id for c in problem.customers[1:] if c.demand <= dcap * 0.5]
        tcand = [c.id for c in problem.customers[1:] if c.id not in dcand]
    def nn_split(cand, n_v, cap):
        routes = [[] for _ in range(n_v)]
        if not cand: return routes
        rem = set(cand)
        for v in range(n_v):
            if not rem: break
            load = 0.0; cur = 0
            while rem:
                nxt = min(rem, key=lambda c: problem.dist[cur][c])
                if load + problem.customers[nxt].demand > cap: break
                routes[v].append(nxt)
                load += problem.customers[nxt].demand
                cur = nxt; rem.remove(nxt)
        if rem and routes: routes[0].extend(list(rem))
        return routes
    return Solution(
        truck_routes=nn_split(tcand, problem.num_trucks, problem.truck_spec.capacity),
        drone_routes=nn_split(dcand, problem.num_drones, dcap))

def random_neighbor(s, problem):
    ops = []
    def nsw(s):
        s = s.copy()
        rs = [r for r in s.truck_routes + s.drone_routes if len(r) >= 2]
        if not rs: return s
        r = random.choice(rs); i, j = random.sample(range(len(r)), 2)
        r[i], r[j] = r[j], r[i]; return s
    def nrel(s):
        s = s.copy()
        rs = s.truck_routes + s.drone_routes
        ne = [i for i, r in enumerate(rs) if r]
        if not ne: return s
        src = rs[random.choice(ne)]
        c = src.pop(random.randrange(len(src)))
        dst = random.choice(rs)
        dst.insert(random.randrange(len(dst) + 1), c); return s
    def nrev(s):
        s = s.copy()
        rs = [r for r in s.truck_routes + s.drone_routes if len(r) >= 2]
        if not rs: return s
        r = random.choice(rs)
        i, j = sorted(random.sample(range(len(r)), 2))
        r[i:j+1] = reversed(r[i:j+1]); return s
    return random.choice([nsw, nrel, nrev])(s)


# ============================================================
# 알고리즘 (간략화, 핵심 동작만)
# ============================================================
class GA:
    name = "StandardGA"
    def __init__(self, problem, max_iter=200, pop_size=50,
                 elitism=2, mutation_rate=0.2, crossover_rate=0.8,
                 tournament_size=3, seed=None):
        self.problem = problem; self.max_iter = max_iter
        self.pop_size = pop_size; self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.seed = seed
        self.history = []
        self.best_solution = None
        self.best_fitness = float("inf")

    def _crossover(self, p1, p2):
        child = p1.copy()
        flat = [("t", c) for r in p2.truck_routes for c in r] + \
               [("d", c) for r in p2.drone_routes for c in r]
        if not flat: return child
        picked = random.sample(flat, len(flat) // 2)
        ps = set(c for _, c in picked)
        for r in child.truck_routes + child.drone_routes:
            r[:] = [c for c in r if c not in ps]
        for kind, c in picked:
            tgt = (random.choice(child.truck_routes) if kind == "t" and child.truck_routes
                   else random.choice(child.drone_routes) if kind == "d" and child.drone_routes
                   else random.choice(child.truck_routes + child.drone_routes))
            tgt.append(c)
        return child

    def _tournament(self, pop, fit):
        idx = random.sample(range(len(pop)), self.tournament_size)
        return pop[min(idx, key=lambda i: fit[i])]

    def run(self):
        if self.seed is not None: random.seed(self.seed)
        pop = [greedy_solution(self.problem)] + [random_solution(self.problem)
                                                  for _ in range(self.pop_size - 1)]
        fit = [self.problem.evaluate(s) for s in pop]
        for _ in range(self.max_iter):
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            new_pop = [pop[i].copy() for i in order[:self.elitism]]
            new_fit = [fit[i] for i in order[:self.elitism]]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fit); p2 = self._tournament(pop, fit)
                child = self._crossover(p1, p2) if random.random() < self.crossover_rate else p1.copy()
                if random.random() < self.mutation_rate:
                    child = random_neighbor(child, self.problem)
                new_pop.append(child); new_fit.append(self.problem.evaluate(child))
            pop, fit = new_pop, new_fit
            bi = min(range(len(pop)), key=lambda i: fit[i])
            if fit[bi] < self.best_fitness:
                self.best_fitness = fit[bi]; self.best_solution = pop[bi].copy()
            self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness


class ElitistGA(GA):
    name = "ElitistGA"
    def __init__(self, problem, **kw):
        kw.setdefault("elitism", 10); kw.setdefault("mutation_rate", 0.05)
        kw.setdefault("crossover_rate", 0.9); kw.setdefault("tournament_size", 5)
        super().__init__(problem, **kw)


class AdaptiveGA(GA):
    name = "AdaptiveGA"
    def __init__(self, problem, **kw):
        super().__init__(problem, **kw)
        self.base_mutation = self.mutation_rate
        self.stagnation = 0; self.last_best = float("inf")
    def run(self):
        if self.seed is not None: random.seed(self.seed)
        pop = [greedy_solution(self.problem)] + [random_solution(self.problem)
                                                  for _ in range(self.pop_size - 1)]
        fit = [self.problem.evaluate(s) for s in pop]
        for _ in range(self.max_iter):
            cb = min(fit)
            if cb >= self.last_best - 1e-6: self.stagnation += 1
            else: self.stagnation = 0
            self.last_best = cb
            self.mutation_rate = min(0.6, self.base_mutation * 2.0) \
                if self.stagnation >= 5 else self.base_mutation
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            new_pop = [pop[i].copy() for i in order[:self.elitism]]
            new_fit = [fit[i] for i in order[:self.elitism]]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fit); p2 = self._tournament(pop, fit)
                child = self._crossover(p1, p2) if random.random() < self.crossover_rate else p1.copy()
                if random.random() < self.mutation_rate:
                    child = random_neighbor(child, self.problem)
                new_pop.append(child); new_fit.append(self.problem.evaluate(child))
            pop, fit = new_pop, new_fit
            bi = min(range(len(pop)), key=lambda i: fit[i])
            if fit[bi] < self.best_fitness:
                self.best_fitness = fit[bi]; self.best_solution = pop[bi].copy()
            self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness


class HS:
    name = "StandardHS"
    def __init__(self, problem, max_iter=1000, hm_size=30,
                 hmcr=0.9, par=0.3, seed=None):
        self.problem = problem; self.max_iter = max_iter
        self.hm_size = hm_size; self.hmcr = hmcr; self.par = par
        self.seed = seed
        self.history = []
        self.best_solution = None
        self.best_fitness = float("inf")
    def run(self):
        if self.seed is not None: random.seed(self.seed)
        hm = [greedy_solution(self.problem)] + [random_solution(self.problem)
                                                 for _ in range(self.hm_size - 1)]
        hm_fit = [self.problem.evaluate(s) for s in hm]
        for it in range(self.max_iter):
            self._update_params(it)
            new_h = self._generate(hm, hm_fit)
            new_fit = self.problem.evaluate(new_h)
            wi = max(range(len(hm)), key=lambda i: hm_fit[i])
            if new_fit < hm_fit[wi]:
                hm[wi] = new_h; hm_fit[wi] = new_fit
            bi = min(range(len(hm)), key=lambda i: hm_fit[i])
            if hm_fit[bi] < self.best_fitness:
                self.best_fitness = hm_fit[bi]; self.best_solution = hm[bi].copy()
            self.history.append(self.best_fitness)
        return self.best_solution, self.best_fitness
    def _update_params(self, it): pass
    def _generate(self, hm, hm_fit):
        if random.random() < self.hmcr:
            h = random.choice(hm).copy()
            if random.random() < self.par: h = random_neighbor(h, self.problem)
            return h
        return random_solution(self.problem)


class ImprovedHS(HS):
    name = "ImprovedHS"
    def __init__(self, problem, par_min=0.1, par_max=0.9, **kw):
        super().__init__(problem, **kw)
        self.par_min = par_min; self.par_max = par_max
    def _update_params(self, it):
        self.par = self.par_min + (self.par_max - self.par_min) * (it / self.max_iter)


class GlobalBestHS(HS):
    name = "GlobalBestHS"
    def _generate(self, hm, hm_fit):
        bi = min(range(len(hm)), key=lambda i: hm_fit[i])
        if random.random() < self.hmcr:
            if random.random() < self.par:
                return random_neighbor(hm[bi].copy(), self.problem)
            return random.choice(hm).copy()
        return random_solution(self.problem)


ALGORITHMS = {
    "StandardGA":   (GA, {"max_iter": 200, "pop_size": 50}),
    "ElitistGA":    (ElitistGA, {"max_iter": 200, "pop_size": 50}),
    "AdaptiveGA":   (AdaptiveGA, {"max_iter": 200, "pop_size": 50}),
    "StandardHS":   (HS, {"max_iter": 1000, "hm_size": 30}),
    "ImprovedHS":   (ImprovedHS, {"max_iter": 1000, "hm_size": 30}),
    "GlobalBestHS": (GlobalBestHS, {"max_iter": 1000, "hm_size": 30}),
}


# ============================================================
# SQLite — 실험 결과 저장
# ============================================================
DB_PATH = "experiments.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, dataset TEXT, algorithm TEXT, run INTEGER,
        fitness REAL, cost REAL, penalty REAL,
        n_trucks INTEGER, n_drones INTEGER, time_sec REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

def save_results(name, dataset, results):
    conn = sqlite3.connect(DB_PATH)
    for r in results:
        conn.execute("""INSERT INTO experiments
            (name, dataset, algorithm, run, fitness, cost, penalty,
             n_trucks, n_drones, time_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, dataset, r["algorithm"], r["run"], r["fitness"],
             r["cost"], r["penalty"], r["n_trucks"], r["n_drones"], r["time_sec"]))
    conn.commit(); conn.close()

def load_experiments():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM experiments ORDER BY created_at DESC", conn)
    conn.close(); return df

init_db()


# ============================================================
# UI 시작
# ============================================================
st.title("🚁 드론-트럭 협력 배송 라우팅 최적화")
st.caption("GA·HS 메타휴리스틱 변종 6종 비교 분석")

# ----- 사이드바: 프로젝트 관리 -----
with st.sidebar:
    # 로그인 상태 표시
    st.markdown(f"👤 **{st.session_state.get('username', 'guest')}** 님")
    if st.button("로그아웃", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()
    st.divider()
    
    st.header("📁 프로젝트")
    project_name = st.text_input("프로젝트명", value="실험_" + time.strftime("%m%d_%H%M"))
    st.divider()
    st.header("📊 과거 실험")
    if st.button("불러오기"):
        df_hist = load_experiments()
        st.dataframe(df_hist[["name", "dataset", "algorithm", "fitness", "created_at"]],
                     hide_index=True, height=300)

# ----- 탭으로 화면 구성 -----
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 데이터", "2️⃣ 파라미터", "3️⃣ 실행", "4️⃣ 결과"])

# ===== 탭 1: 데이터 =====
with tab1:
    st.subheader("Solomon VRPTW 데이터 업로드")
    col1, col2 = st.columns([1, 2])
    with col1:
        data_source = st.radio("데이터 소스", ["파일 업로드", "예제 데이터"])
        if data_source == "파일 업로드":
            uploaded = st.file_uploader("Solomon .txt 파일", type=["txt"])
            if uploaded:
                st.session_state["data_text"] = uploaded.read().decode("utf-8")
                st.session_state["data_name"] = uploaded.name
                st.success(f"✅ {uploaded.name} 로드 완료")
        else:
            example = st.selectbox("예제 선택",
                ["C101 (Clustered)", "R101 (Random)", "RC101 (Mixed)"])
            st.info("💡 예제 데이터는 별도 파일로 같이 제공됩니다 (C101.txt 등)")
    
    if "data_text" in st.session_state:
        with col2:
            try:
                p = VRPTWProblem.from_solomon_text(st.session_state["data_text"])
                st.session_state["preview_problem"] = p
                fig, ax = plt.subplots(figsize=(6, 5))
                xs = [c.x for c in p.customers]; ys = [c.y for c in p.customers]
                ax.scatter(xs[1:], ys[1:], c="gray", s=30, label="Customer")
                ax.scatter([xs[0]], [ys[0]], c="black", s=200, marker="s", label="Depot")
                ax.set_title(f"Customer Locations (n={p.n_customers})")
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("고객 수", p.n_customers)
                c2.metric("총 수요", f"{sum(c.demand for c in p.customers[1:]):.0f}")
                c3.metric("트럭 용량", f"{p.truck_spec.capacity:.0f}")
            except Exception as e:
                st.error(f"파일 파싱 실패: {e}")

# ===== 탭 2: 파라미터 =====
with tab2:
    st.subheader("차량 및 알고리즘 파라미터")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚚 차량 운용**")
        num_trucks = st.slider("트럭 대수", 1, 30, 12)
        num_drones = st.slider("드론 대수", 1, 20, 8)
        st.markdown("**🚁 드론 사양**")
        drone_cap_ratio = st.slider("드론 용량 비율 (트럭 대비)", 0.05, 0.5, 0.15, 0.05)
        drone_speed = st.slider("드론 속도 (트럭 대비)", 1.0, 4.0, 2.0, 0.5)
        drone_max_dist = st.slider("드론 최대 거리", 20, 150, 50, 5)
    with col2:
        st.markdown("**⚖️ 페널티 가중치**")
        tw_w = st.number_input("시간창 위반 가중치", value=10.0)
        cap_w = st.number_input("용량 초과 가중치", value=5000.0)
        dist_w = st.number_input("배터리 초과 가중치", value=5000.0)
        st.session_state["params"] = dict(
            num_trucks=num_trucks, num_drones=num_drones,
            drone_cap_ratio=drone_cap_ratio, drone_speed=drone_speed,
            drone_max_dist=drone_max_dist,
            tw_w=tw_w, cap_w=cap_w, dist_w=dist_w)

# ===== 탭 3: 실행 =====
with tab3:
    st.subheader("알고리즘 실행")
    if "data_text" not in st.session_state:
        st.warning("⚠️ 먼저 탭 1에서 데이터를 업로드해주세요.")
    else:
        selected = st.multiselect("비교할 알고리즘 선택",
            list(ALGORITHMS.keys()), default=list(ALGORITHMS.keys()))
        n_runs = st.slider("알고리즘당 반복 횟수", 1, 10, 3)
        
        if st.button("🚀 최적화 실행", type="primary"):
            params = st.session_state.get("params", {})
            problem = VRPTWProblem.from_solomon_text(
                st.session_state["data_text"],
                num_trucks=params.get("num_trucks", 12),
                num_drones=params.get("num_drones", 8),
                drone_cap_ratio=params.get("drone_cap_ratio", 0.15),
                drone_speed=params.get("drone_speed", 2.0),
                drone_max_dist=params.get("drone_max_dist", 50.0))
            problem.tw_penalty = params.get("tw_w", 10.0)
            problem.cap_penalty = params.get("cap_w", 5000.0)
            problem.dist_penalty = params.get("dist_w", 5000.0)
            
            results = []
            total = len(selected) * n_runs
            prog = st.progress(0.0, text="실행 중...")
            done = 0
            
            for algo_name in selected:
                AlgoCls, kw = ALGORITHMS[algo_name]
                for r in range(n_runs):
                    algo = AlgoCls(problem, seed=r, **kw)
                    t0 = time.time()
                    sol, fit = algo.run()
                    elapsed = time.time() - t0
                    bd = problem.evaluate(sol, return_breakdown=True)
                    results.append(dict(
                        algorithm=algo_name, run=r,
                        fitness=fit, cost=bd["cost"], penalty=bd["penalty"],
                        n_trucks=bd["n_trucks"], n_drones=bd["n_drones"],
                        truck_dist=bd["truck_distance"],
                        drone_dist=bd["drone_distance"],
                        time_sec=elapsed, history=algo.history, solution=sol))
                    done += 1
                    prog.progress(done / total, text=f"{algo_name} run {r+1}/{n_runs}")
            
            prog.empty()
            st.session_state["results"] = results
            st.session_state["problem"] = problem
            save_results(project_name,
                         st.session_state.get("data_name", "unknown"),
                         results)
            st.success(f"✅ 총 {len(results)}개 실험 완료. 결과 탭에서 확인하세요.")

# ===== 탭 4: 결과 =====
with tab4:
    st.subheader("실험 결과 분석")
    if "results" not in st.session_state:
        st.warning("⚠️ 먼저 탭 3에서 실험을 실행해주세요.")
    else:
        results = st.session_state["results"]
        problem = st.session_state["problem"]
        df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("history", "solution")} for r in results])
        
        # 요약 표
        st.markdown("**📊 알고리즘별 요약**")
        summary = df.groupby("algorithm").agg(
            mean_fitness=("fitness", "mean"),
            std_fitness=("fitness", "std"),
            min_fitness=("fitness", "min"),
            mean_time=("time_sec", "mean"),
            mean_trucks=("n_trucks", "mean"),
            mean_drones=("n_drones", "mean")).round(2)
        st.dataframe(summary)
        
        # 다운로드
        c1, c2 = st.columns(2)
        c1.download_button("📥 요약 CSV", summary.to_csv(),
                          "summary.csv", "text/csv")
        c2.download_button("📥 전체 CSV", df.to_csv(index=False),
                          "results_raw.csv", "text/csv")
        
        st.divider()
        # 박스플롯
        st.markdown("**📦 fitness 분포 (박스플롯)**")
        fig1, ax = plt.subplots(figsize=(8, 5))
        algos = df.algorithm.unique()
        data_list = [df[df.algorithm == a]["fitness"].values for a in algos]
        ax.boxplot(data_list, labels=algos)
        ax.set_yscale("log")
        ax.set_ylabel("Fitness (log scale, lower=better)")
        ax.grid(alpha=0.3); plt.xticks(rotation=20)
        st.pyplot(fig1); plt.close(fig1)
        
        # 수렴곡선
        st.markdown("**📈 수렴 곡선**")
        fig2, ax = plt.subplots(figsize=(8, 5))
        for algo_name in algos:
            sample = next(r for r in results if r["algorithm"] == algo_name and r["run"] == 0)
            ax.plot(sample["history"], label=algo_name, linewidth=1.5)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness")
        ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig2); plt.close(fig2)
        
        # 최적 경로
        st.markdown("**🗺️ 최적 경로 시각화**")
        best = min(results, key=lambda r: r["fitness"])
        sol = best["solution"]
        fig3, ax = plt.subplots(figsize=(8, 7))
        xs = [c.x for c in problem.customers]; ys = [c.y for c in problem.customers]
        ax.scatter(xs[1:], ys[1:], c="gray", s=30, zorder=2)
        ax.scatter([xs[0]], [ys[0]], c="black", s=200, marker="s", label="Depot", zorder=3)
        tc = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(sol.truck_routes), 1)))
        for i, r in enumerate(sol.truck_routes):
            if not r: continue
            path = [0] + r + [0]
            ax.plot([problem.customers[p].x for p in path],
                    [problem.customers[p].y for p in path],
                    "-", color=tc[i], linewidth=2,
                    label=f"Truck {i+1}" if i < 3 else None)
        dc = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(sol.drone_routes), 1)))
        for i, r in enumerate(sol.drone_routes):
            if not r: continue
            path = [0] + r + [0]
            ax.plot([problem.customers[p].x for p in path],
                    [problem.customers[p].y for p in path],
                    "--", color=dc[i], linewidth=1.5,
                    label=f"Drone {i+1}" if i < 3 else None)
        ax.set_title(f"Best: {best['algorithm']}, fitness={best['fitness']:.1f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig3); plt.close(fig3)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("최저 fitness", f"{best['fitness']:.1f}")
        c2.metric("사용 트럭", f"{best['n_trucks']}대")
        c3.metric("사용 드론", f"{best['n_drones']}대")

        # ============================================================
        # 트럭 단독 vs 트럭+드론 비교 분석 (Baseline 비교)
        # ============================================================
        st.divider()
        st.markdown("### 🆚 트럭 단독 vs 트럭+드론 비교")
        st.caption("드론 도입 효과를 정량적으로 분석합니다")

        if st.button("⚖️ Baseline 비교 실행", type="secondary"):
            with st.spinner("트럭 단독 시나리오 최적화 중..."):
                # 트럭 단독 문제 생성 (드론 0대)
                params = st.session_state.get("params", {})
                baseline_problem = VRPTWProblem.from_solomon_text(
                    st.session_state["data_text"],
                    num_trucks=params.get("num_trucks", 12) + params.get("num_drones", 8),
                    num_drones=0,
                    drone_cap_ratio=params.get("drone_cap_ratio", 0.15),
                    drone_speed=params.get("drone_speed", 2.0),
                    drone_max_dist=params.get("drone_max_dist", 50.0))
                baseline_problem.tw_penalty = params.get("tw_w", 10.0)
                baseline_problem.cap_penalty = params.get("cap_w", 5000.0)
                baseline_problem.dist_penalty = params.get("dist_w", 5000.0)

                # 최고 성능 알고리즘으로 baseline 풀기 (AdaptiveGA)
                AlgoCls, kw = ALGORITHMS["AdaptiveGA"]
                baseline_runs = []
                for r in range(3):  # 빠르게 3회만
                    algo = AlgoCls(baseline_problem, seed=r, **kw)
                    t0 = time.time()
                    sol_b, fit_b = algo.run()
                    bd_b = baseline_problem.evaluate(sol_b, return_breakdown=True)
                    baseline_runs.append({
                        "fitness": fit_b, "cost": bd_b["cost"],
                        "truck_distance": bd_b["truck_distance"],
                        "n_trucks": bd_b["n_trucks"],
                        "solution": sol_b, "time": time.time() - t0
                    })

                baseline_best = min(baseline_runs, key=lambda r: r["fitness"])
                hybrid_best = min((r for r in results if "GA" in r["algorithm"]),
                                  key=lambda r: r["fitness"])

                # 비교 표
                st.markdown("**📊 시나리오별 성능 비교**")
                comparison = pd.DataFrame({
                    "지표": ["적합도 (fitness)", "운영 비용 (cost)",
                             "트럭 이동거리", "드론 이동거리",
                             "사용 트럭 수", "사용 드론 수"],
                    "🚚 트럭 단독": [
                        f"{baseline_best['fitness']:,.1f}",
                        f"{baseline_best['cost']:,.1f}",
                        f"{baseline_best['truck_distance']:,.1f}",
                        "—",
                        f"{baseline_best['n_trucks']}대",
                        "0대"
                    ],
                    "🚁 트럭+드론": [
                        f"{hybrid_best['fitness']:,.1f}",
                        f"{hybrid_best['cost']:,.1f}",
                        f"{hybrid_best.get('truck_dist', 0):,.1f}" if 'truck_dist' in hybrid_best else "—",
                        f"{hybrid_best.get('drone_dist', 0):,.1f}" if 'drone_dist' in hybrid_best else "—",
                        f"{hybrid_best['n_trucks']}대",
                        f"{hybrid_best['n_drones']}대"
                    ],
                })
                st.dataframe(comparison, hide_index=True, use_container_width=True)

                # 핵심 지표 (비용 절감률)
                cost_saving = baseline_best['cost'] - hybrid_best['cost']
                cost_saving_pct = (cost_saving / baseline_best['cost']) * 100 if baseline_best['cost'] > 0 else 0
                fit_saving = baseline_best['fitness'] - hybrid_best['fitness']
                fit_saving_pct = (fit_saving / baseline_best['fitness']) * 100 if baseline_best['fitness'] > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "💰 운영비용 절감",
                    f"{cost_saving:,.1f}",
                    f"{cost_saving_pct:+.1f}%",
                    delta_color="inverse" if cost_saving > 0 else "normal"
                )
                col2.metric(
                    "🎯 적합도 개선",
                    f"{fit_saving:,.1f}",
                    f"{fit_saving_pct:+.1f}%",
                    delta_color="inverse" if fit_saving > 0 else "normal"
                )
                col3.metric(
                    "🚁 드론 분담률",
                    f"{(hybrid_best.get('drone_dist', 0) / (hybrid_best.get('truck_dist', 1) + hybrid_best.get('drone_dist', 0)) * 100):.1f}%"
                    if 'drone_dist' in hybrid_best else "—"
                )

                # 결론 인사이트
                if cost_saving_pct > 5:
                    st.success(
                        f"✅ **드론 도입 효과 입증**: 트럭 단독 대비 운영비용 "
                        f"**{cost_saving_pct:.1f}% 절감** 효과 확인. "
                        f"드론이 전체 이동거리의 "
                        f"{(hybrid_best.get('drone_dist', 0) / (hybrid_best.get('truck_dist', 1) + hybrid_best.get('drone_dist', 0)) * 100):.1f}%"
                        f"를 분담하여 트럭 부담을 경감했습니다."
                    )
                elif cost_saving_pct > 0:
                    st.info(
                        f"ℹ️ **드론 도입 효과 제한적**: 비용 절감은 있으나 "
                        f"({cost_saving_pct:.1f}%) 미미한 수준입니다. "
                        f"드론 사양 또는 운용 정책 재검토 필요."
                    )
                else:
                    st.warning(
                        f"⚠️ **드론 도입 효과 불확실**: 본 시나리오에서는 "
                        f"트럭 단독 운용이 더 효율적일 수 있습니다. "
                        f"드론 사양/페널티 가중치 조정 필요."
                    )

                # 시각적 비교 그래프
                st.markdown("**📊 비용 구성 비교**")
                fig_cmp, axes = plt.subplots(1, 2, figsize=(12, 5))

                # 좌: 비용 막대그래프
                axes[0].bar(["Truck Only", "Truck + Drone"],
                            [baseline_best['cost'], hybrid_best['cost']],
                            color=['#3498db', '#2ecc71'])
                axes[0].set_ylabel("Operating Cost")
                axes[0].set_title("Operating Cost Comparison")
                axes[0].grid(alpha=0.3, axis='y')
                for i, v in enumerate([baseline_best['cost'], hybrid_best['cost']]):
                    axes[0].text(i, v, f"{v:,.0f}", ha='center', va='bottom')

                # 우: 이동거리 stacked bar
                truck_dist_b = baseline_best['truck_distance']
                truck_dist_h = hybrid_best.get('truck_dist', 0)
                drone_dist_h = hybrid_best.get('drone_dist', 0)
                axes[1].bar(["Truck Only", "Truck + Drone"],
                            [truck_dist_b, truck_dist_h],
                            color='#3498db', label='Truck')
                axes[1].bar(["Truck Only", "Truck + Drone"],
                            [0, drone_dist_h],
                            bottom=[0, truck_dist_h],
                            color='#e74c3c', label='Drone')
                axes[1].set_ylabel("Total Distance")
                axes[1].set_title("Distance Composition")
                axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

                plt.tight_layout()
                st.pyplot(fig_cmp); plt.close(fig_cmp)