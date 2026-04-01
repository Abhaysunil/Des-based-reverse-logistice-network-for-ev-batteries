import simpy
import random
import statistics
import matplotlib.pyplot as plt
from datafile import battery_data

# ---------------- PARAMETERS ----------------
PARAMS = {
    
    "arrival_rate": 5,

    "inspection_mean": 10,
    "inspection_std": 3,

    "soh_threshold_reuse": 80,
    "soh_threshold_reman": 60,

    "misclassification_rate": 0.05,

    "cost_per_km": 2,
    "distance_range": (10, 50),

    "reuse_revenue": 120,
    "reman_revenue": 90,
    "recycle_revenue": 50,

    "reuse_cost": 50,
    "reman_cost": 70,
    "recycle_cost": 40,

    "inspection_cost": 10
}

random.seed(42)

# ---------------- SIMULATION FUNCTION ----------------
def run_simulation(use_classification=True):
    total_costs = []
    total_times = []
    waiting_times = []
    profits = []

    routes = {"Reuse": 0, "Reman": 0, "Recycle": 0}

    env = simpy.Environment()

    inspection = simpy.Resource(env, capacity=2)
    reuse = simpy.Resource(env, capacity=1)
    reman = simpy.Resource(env, capacity=2)
    recycle = simpy.Resource(env, capacity=3)

    def battery_process(env, battery):
        start_time = env.now
        cost = 0
        revenue = 0

        soh = battery["SOH"] * 100

        # ---- TRANSPORT ----
        distance = random.uniform(*PARAMS["distance_range"])
        yield env.timeout(distance * 2)
        cost += distance * PARAMS["cost_per_km"]

        # ---- INSPECTION ----
        with inspection.request() as req:
            yield req
            inspection_time = max(
                1,
                random.gauss(PARAMS["inspection_mean"], PARAMS["inspection_std"])
            )
            yield env.timeout(inspection_time)
            cost += PARAMS["inspection_cost"]

        # ---- CLASSIFICATION ----
        if use_classification:
            if soh >= PARAMS["soh_threshold_reuse"]:
                route = "Reuse"
            elif soh >= PARAMS["soh_threshold_reman"]:
                route = "Reman"
            else:
                route = "Recycle"

            # Misclassification applies only when classification exists
            if random.random() < PARAMS["misclassification_rate"]:
                route = random.choice(["Reuse", "Reman", "Recycle"])
        else:
            # Base case: no classification, direct to recycling
            route = "Recycle"

        routes[route] += 1

        # ---- PROCESSING ----
        if route == "Reuse":
            with reuse.request() as req:
                wait_start = env.now
                yield req
                waiting_times.append(env.now - wait_start)

                yield env.timeout(random.uniform(20, 40))
                cost += PARAMS["reuse_cost"]
                revenue += PARAMS["reuse_revenue"]

        elif route == "Reman":
            with reman.request() as req:
                wait_start = env.now
                yield req
                waiting_times.append(env.now - wait_start)

                yield env.timeout(random.uniform(30, 60))
                cost += PARAMS["reman_cost"]
                revenue += PARAMS["reman_revenue"]

        else:
            with recycle.request() as req:
                wait_start = env.now
                yield req
                waiting_times.append(env.now - wait_start)

                yield env.timeout(random.uniform(40, 80))
                cost += PARAMS["recycle_cost"]
                revenue += PARAMS["recycle_revenue"]

        total_costs.append(cost)
        total_times.append(env.now - start_time)
        profits.append(revenue - cost)

    def arrival(env):
        for b in battery_data:
            yield env.timeout(random.expovariate(1.0 / PARAMS["arrival_rate"]))
            env.process(battery_process(env, b))

    env.process(arrival(env))
    env.run()

    return {
        "cost": statistics.mean(total_costs),
        "time": statistics.mean(total_times),
        "wait": statistics.mean(waiting_times),
        "profit": statistics.mean(profits),
        "routes": routes
    }

# ---------------- BASE vs PROPOSED ----------------
base = run_simulation(use_classification=False)
proposed = run_simulation(use_classification=True)

print("\n===== BASE CASE =====")
print(base)

print("\n===== PROPOSED MODEL =====")
print(proposed)

# ---------------- SENSITIVITY ANALYSIS ----------------
def sensitivity_analysis():
    thresholds = [75, 78, 80, 82, 85]
    results = []

    original_threshold = PARAMS["soh_threshold_reuse"]

    for t in thresholds:
        PARAMS["soh_threshold_reuse"] = t
        res = run_simulation(use_classification=True)

        results.append({
            "threshold": t,
            "profit": res["profit"],
            "cost": res["cost"]
        })

    # restore original threshold
    PARAMS["soh_threshold_reuse"] = original_threshold

    return results

sens_results = sensitivity_analysis()

print("\n===== SENSITIVITY ANALYSIS =====")
for r in sens_results:
    print(r)

# ---------------- GRAPHS ----------------
labels = ["Cost", "Time", "Waiting", "Profit"]
base_vals = [base["cost"], base["time"], base["wait"], base["profit"]]
prop_vals = [proposed["cost"], proposed["time"], proposed["wait"], proposed["profit"]]

x = range(len(labels))

plt.figure(figsize=(8, 5))
plt.bar(x, base_vals)
plt.xticks(x, labels)
plt.title("Base Case Performance")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(x, prop_vals)
plt.xticks(x, labels)
plt.title("Proposed Model Performance")
plt.ylabel("Value")
plt.show()

thresholds = [r["threshold"] for r in sens_results]
profits = [r["profit"] for r in sens_results]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, profits, marker='o')
plt.xlabel("SOH Threshold")
plt.ylabel("Profit")
plt.title("Sensitivity Analysis")
plt.grid(True)
plt.show()
