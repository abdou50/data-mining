import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# =====================================================
# STEP 1: LOAD DATASET
# =====================================================
df = pd.read_csv("Mall_Customers.csv")

print("\n================ STEP 1 ================")
print("Dataset loaded successfully")
print(df.head())

print("""
شرح:
هنا قرينا dataset.
كل ligne = client.
عنا Age, Gender, Annual Income, Spending Score.
""")


# =====================================================
# STEP 2: VISUALISE RAW DATA
# =====================================================
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    s=70,
    edgecolor="black"
)
plt.title("Step 2: Raw Data Before DBSCAN")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

print("\n================ STEP 2 ================")
print("""
شرح:
هنا نشوفو data قبل clustering.
كل point = client.
X = income
Y = spending score

الهدف:
نلقاو groupes متاع clients قريبين لبعضهم.
""")


# =====================================================
# STEP 3: SELECT FEATURES
# =====================================================
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\n================ STEP 3 ================")
print("Selected features:")
print(X.head())

print("""
شرح:
اخترنا زوز variables:
1) Annual Income
2) Spending Score

علاش؟
خاطر في Data Mining نحب نفهم comportement client:
شكون دخلو كبير ويصرف برشا؟
شكون دخلو صغير ويصرف قليل؟
إلخ...
""")


# =====================================================
# STEP 4: STANDARDIZATION
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df["Income_scaled"] = X_scaled[:, 0]
df["Spending_scaled"] = X_scaled[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(
    df["Income_scaled"],
    df["Spending_scaled"],
    s=70,
    edgecolor="black"
)
plt.title("Step 4: Data After Scaling")
plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.grid(True)
plt.show()

print("\n================ STEP 4 ================")
print("""
شرح:
DBSCAN يخدم بالمسافة بين points.
لازم نعملو scaling باش variables يكونو في نفس niveau.

يعني:
Income و Spending Score ولى عندهم نفس poids في الحساب.
""")


# =====================================================
# STEP 5: K-DISTANCE GRAPH
# =====================================================
min_samples = 5

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = sorted(distances[:, min_samples - 1])

plt.figure(figsize=(9, 6))
plt.plot(distances)
plt.title("Step 5: K-distance Graph pour choisir eps")
plt.xlabel("Points triés")
plt.ylabel(f"Distance au {min_samples}ème voisin")
plt.grid(True)
plt.show()

print("\n================ STEP 5 ================")
print("""
شرح:
هذا graph يعاوننا نختارو eps.

كيفاش؟
نشوفو وين curve تبدأ تطلع بقوة.
النقطة هذي اسمها elbow / knee.

في graph متاعك تقريبا eps بين:
0.4 و 0.6

نبدأو بـ eps = 0.5
""")


# =====================================================
# STEP 6: APPLY DBSCAN
# =====================================================
eps = 0.5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df["Cluster"] = dbscan.fit_predict(X_scaled)

print("\n================ STEP 6 ================")
print("DBSCAN applied")
print("eps =", eps)
print("min_samples =", min_samples)

print("\nCluster counts:")
print(df["Cluster"].value_counts().sort_index())

print("""
شرح:
DBSCAN عطانا labels.

Cluster 0 = groupe
Cluster 1 = groupe
Cluster 2 = groupe
...
Cluster -1 = Noise / Outlier

يعني -1 client موش تابع حتى groupe.
""")


# =====================================================
# STEP 7: VISUALISE FINAL CLUSTERS
# =====================================================
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="tab10",
    s=80,
    edgecolor="black"
)
plt.title("Step 7: DBSCAN Final Clusters")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

print("\n================ STEP 7 ================")
print("""
شرح:
هنا كل couleur تمثل cluster.

النقاط اللي عندها نفس اللون:
عندهم comportement قريب.

DBSCAN ما طلبش منا عدد clusters.
هو وحدو لقاهم حسب density.
""")


# =====================================================
# STEP 8: VISUALISE NOISE ONLY
# =====================================================
normal_points = df[df["Cluster"] != -1]
noise_points = df[df["Cluster"] == -1]

plt.figure(figsize=(8, 6))

plt.scatter(
    normal_points["Annual Income (k$)"],
    normal_points["Spending Score (1-100)"],
    c="lightgray",
    s=70,
    edgecolor="black",
    label="Clustered points"
)

plt.scatter(
    noise_points["Annual Income (k$)"],
    noise_points["Spending Score (1-100)"],
    c="red",
    s=120,
    edgecolor="black",
    label="Noise / Outliers"
)

plt.title("Step 8: Noise Points Detected by DBSCAN")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

print("\n================ STEP 8 ================")
print("""
شرح:
النقاط الحمراء = Noise.

Noise معناها:
client بعيد على بقية clients
وما عندوش enough neighbors داخل eps.

في DBSCAN:
Noise = Cluster -1
""")


# =====================================================
# STEP 9: CORE POINTS / BORDER / NOISE
# =====================================================
core_samples = dbscan.core_sample_indices_

df["Point_Type"] = "Border"
df.loc[core_samples, "Point_Type"] = "Core"
df.loc[df["Cluster"] == -1, "Point_Type"] = "Noise"

colors = {
    "Core": "green",
    "Border": "orange",
    "Noise": "red"
}

plt.figure(figsize=(8, 6))

for point_type in ["Core", "Border", "Noise"]:
    subset = df[df["Point_Type"] == point_type]
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        c=colors[point_type],
        s=80,
        edgecolor="black",
        label=point_type
    )

plt.title("Step 9: Core / Border / Noise Points")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

print("\n================ STEP 9 ================")
print(df["Point_Type"].value_counts())

print("""
شرح:
DBSCAN يقسم points إلى 3 types:

1) Core point:
عندو على الأقل min_samples داخل eps.

2) Border point:
موش core، أما قريب من core point.

3) Noise:
بعيد وماهوش تابع حتى cluster.

هذي أهم فكرة في DBSCAN.
""")


# =====================================================
# STEP 10: COMPARE EPS VALUES
# =====================================================
eps_values = [0.3, 0.4, 0.5, 0.6]

for e in eps_values:
    model = DBSCAN(eps=e, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=clusters,
        cmap="tab10",
        s=80,
        edgecolor="black"
    )
    plt.title(f"Step 10: DBSCAN with eps = {e}")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.show()

    print(f"\nResult for eps = {e}")
    print(pd.Series(clusters).value_counts().sort_index())

print("""
شرح:
هنا جربنا plusieurs valeurs متاع eps.

eps صغير:
برشا points يخرجو Noise.

eps كبير:
clusters يندمجو مع بعضهم.

أحسن eps هو اللي يعطي clusters واضحين و noise معقول.
""")


# =====================================================
# STEP 11: CLUSTER SUMMARY
# =====================================================
summary = df.groupby("Cluster")[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

print("\n================ STEP 11 ================")
print("Cluster summary:")
print(summary)

print("""
شرح:
هذا tableau يفسر كل cluster.

مثلا:
Cluster فيه income عالي و spending عالي
= clients importants.

Cluster فيه income قليل و spending قليل
= clients économiques.

Cluster -1
= clients atypiques / outliers.
""")


# =====================================================
# STEP 12: SAVE FINAL RESULT
# =====================================================
df.to_csv("Mall_Customers_DBSCAN_Final_Result.csv", index=False)

print("\n================ STEP 12 ================")
print("Final result saved as Mall_Customers_DBSCAN_Final_Result.csv")

print("""
خلاصة:
طبقنا DBSCAN في Data Mining على dataset متاع clients.

الخطوات:
1) قراءة البيانات
2) visualisation
3) اختيار variables
4) scaling
5) اختيار eps بال K-distance graph
6) تطبيق DBSCAN
7) استخراج clusters
8) استخراج noise
9) تفسير core / border / noise
10) مقارنة eps
11) تفسير النتائج
""")