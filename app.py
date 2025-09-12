import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

# --- Cấu hình trang và tiêu đề ---
st.set_page_config(page_title="Đồ Án Tốt Nghiệp - Project 2", layout="wide", page_icon="🏨")

# --- Các hàm tải và xử lý dữ liệu (với cache để tối ưu hiệu suất) ---

@st.cache_data
def load_data():
    """Tải dữ liệu từ file CSV một cách an toàn và cache lại."""
    def read_csv_safely(path, **kwargs):
        for enc in ["utf-8", "utf-8-sig", "cp1258", "latin-1"]:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except Exception:
                pass
        raise ValueError(f"Không thể đọc file {path} với các encoding đã thử.")

    data_dir = Path("./data/")
    hotel_info = read_csv_safely(data_dir / "hotel_info.csv")
    
    # Xác định các cột quan trọng một cách linh hoạt
    id_cols = [c for c in hotel_info.columns if "Hotel_ID" in c]
    name_cols = [c for c in hotel_info.columns if "Hotel_Name" in c]
    
    hotel_id_col = id_cols[0] if id_cols else hotel_info.columns[0]
    hotel_name_col = name_cols[0] if name_cols else hotel_info.columns[1]
    
    return hotel_info, hotel_id_col, hotel_name_col

@st.cache_resource
def preprocess_and_tfidf(hotel_info, hotel_id_col, hotel_name_col):
    """Tiền xử lý văn bản và tính toán ma trận tương đồng TF-IDF."""
    def simple_clean(s):
        if not isinstance(s, str): s = str(s)
        s = unicodedata.normalize("NFC", s.lower())
        s = re.sub(r"http\S+|www\S+", " ", s)
        s = re.sub(r"[\w.-]+@[\w.-]+", " ", s)
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"[^\w\sáàảãạăằắẳẵặâầấẩẫậéèẻẽẹêềếểễệíìỉĩịóòỏõọôồốổỗộơờớởỡợúùủũụưừứửữựýỳỷỹỵđ]", " ", s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    text_cols = [c for c in hotel_info.columns if "Description" in c]
    hotel_info["_text_info"] = (hotel_info[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else "")
    corpus = hotel_info["_text_info"].map(simple_clean).fillna("")
    
    tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(corpus)
    cos_sim_matrix = cosine_similarity(X)
    return cos_sim_matrix

@st.cache_resource
def load_embeddings(model_name):
    """Tải các file embedding đã được tính toán trước."""
    models_dir = Path("./output/models")
    try:
        embeddings = np.load(models_dir / f"{model_name}_emb.npy")
        return embeddings
    except FileNotFoundError:
        return None

def get_recommendations(sim_matrix, hotel_df, selected_hotel_id, id_col, name_col, top_n=10):
    """Lấy danh sách gợi ý từ ma trận tương đồng."""
    try:
        idx = hotel_df.index[hotel_df[id_col] == selected_hotel_id][0]
        sim_scores = sim_matrix[idx]
        
        top_indices = np.argsort(-sim_scores)[:top_n + 1]
        
        recs = hotel_df.iloc[top_indices][[id_col, name_col]].copy()
        recs["similarity"] = sim_scores[top_indices]
        
        # Loại bỏ chính khách sạn đã chọn và lấy top N
        recs = recs[recs[id_col] != selected_hotel_id].head(top_n)
        recs.rename(columns={"similarity": "Độ tương đồng"}, inplace=True)
        return recs
    except (IndexError, KeyError):
        return pd.DataFrame()

# --- Giao diện chính của ứng dụng ---

st.title("🎓 Đồ Án Tốt Nghiệp - Project 2")
st.subheader("🏨 Hệ thống Gợi ý Khách sạn Agoda")
st.markdown("Chọn một phương pháp và một khách sạn để xem các gợi ý tương tự.")

# Tải dữ liệu
hotel_info, HOTEL_ID_COL, HOTEL_NAME_COL = load_data()

# --- Sidebar cho các tùy chọn ---
with st.sidebar:
    st.image("logo.png", width=100) # Thêm logo (đảm bảo file logo.png ở cùng thư mục)
    st.title("⚙️ Tùy chọn")
    method = st.selectbox(
        "Chọn phương pháp gợi ý:",
        ["TF-IDF", "Doc2Vec", "SBERT", "ALS"]
    )

# --- Khu vực chính ---
st.header("1. Chọn khách sạn bạn thích")
hotel_list = hotel_info[HOTEL_NAME_COL].astype(str).unique().tolist()
seed_hotel_name = st.selectbox("", hotel_list)

# Hiển thị thông tin khách sạn đã chọn
if seed_hotel_name:
    selected_hotel_info = hotel_info[hotel_info[HOTEL_NAME_COL] == seed_hotel_name].iloc[0]
    seed_hotel_id = selected_hotel_info[HOTEL_ID_COL]
    
    with st.expander("Thông tin khách sạn đã chọn"):
        st.subheader(selected_hotel_info[HOTEL_NAME_COL])
        if 'Hotel_Rank' in selected_hotel_info and pd.notna(selected_hotel_info['Hotel_Rank']):
            st.write(f"**Hạng:** {selected_hotel_info['Hotel_Rank']}")
        if 'Hotel_Address' in selected_hotel_info and pd.notna(selected_hotel_info['Hotel_Address']):
            st.write(f"**Địa chỉ:** {selected_hotel_info['Hotel_Address']}")
        if '_text_info' in selected_hotel_info and selected_hotel_info['_text_info']:
            st.caption(selected_hotel_info['_text_info'][:300] + "...")

st.header("2. Kết quả đề xuất")

# --- Logic xử lý và hiển thị kết quả ---
if method == "TF-IDF":
    with st.spinner("Đang tính toán độ tương đồng TF-IDF..."):
        cos_sim_matrix = preprocess_and_tfidf(hotel_info, HOTEL_ID_COL, HOTEL_NAME_COL)
        recommendations = get_recommendations(cos_sim_matrix, hotel_info, seed_hotel_id, HOTEL_ID_COL, HOTEL_NAME_COL)
    st.success(f"Top 10 khách sạn tương tự **{seed_hotel_name}** theo TF-IDF:")
    st.dataframe(recommendations, use_container_width=True)

elif method == "Doc2Vec":
    with st.spinner("Đang tải embedding Doc2Vec và tìm gợi ý..."):
        d2v_emb = load_embeddings("d2v")
        if d2v_emb is not None:
            sim_scores = cosine_similarity([d2v_emb[hotel_info.index[hotel_info[HOTEL_ID_COL] == seed_hotel_id][0]]], d2v_emb)
            recommendations = get_recommendations(sim_scores, hotel_info, seed_hotel_id, HOTEL_ID_COL, HOTEL_NAME_COL)
            st.success(f"Top 10 khách sạn tương tự **{seed_hotel_name}** theo Doc2Vec:")
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.warning("Mô hình Doc2Vec chưa sẵn sàng. Vui lòng chạy notebook để tạo file embedding.")

elif method == "SBERT":
    with st.spinner("Đang tải embedding SBERT và tìm gợi ý..."):
        sbert_emb = load_embeddings("sbert")
        if sbert_emb is not None:
            sim_scores = cosine_similarity([sbert_emb[hotel_info.index[hotel_info[HOTEL_ID_COL] == seed_hotel_id][0]]], sbert_emb)
            recommendations = get_recommendations(sim_scores, hotel_info, seed_hotel_id, HOTEL_ID_COL, HOTEL_NAME_COL)
            st.success(f"Top 10 khách sạn tương tự **{seed_hotel_name}** theo SBERT:")
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.warning("Mô hình SBERT chưa sẵn sàng. Vui lòng chạy notebook để tạo file embedding.")

elif method == "ALS":
    st.subheader("Nhập User ID để nhận gợi ý:")
    user_id = st.number_input("User ID:", min_value=0, step=1, value=0)

    if st.button("Lấy gợi ý cho User"):
        with st.spinner("Đang khởi tạo Spark và tải mô hình ALS..."):
            try:
                import findspark
                findspark.init()
                from pyspark.sql import SparkSession
                from pyspark.ml.recommendation import ALSModel

                spark = SparkSession.builder \
                    .appName("AgodaALSInference") \
                    .master("local[*]") \
                    .getOrCreate()
                
                model_path = str(Path("./output/models/best_als_model"))
                model = ALSModel.load(model_path)
                
                user_df = spark.createDataFrame([(user_id,)], ["userId"])
                recs = model.recommendForUserSubset(user_df, 10).toPandas()
                
                if recs.empty:
                    st.info(f"Không có gợi ý nào cho User ID: {user_id}")
                else:
                    rows = []
                    for item in recs["recommendations"].iloc[0]:
                        rows.append({"itemId": item["itemId"], "score": float(item["rating"])})
                    
                    out_df = pd.DataFrame(rows)
                    
                    # Map itemId về thông tin khách sạn
                    id_map = hotel_info[[HOTEL_ID_COL]].reset_index().rename(columns={'index': 'itemId'})
                    id_map[HOTEL_ID_COL] = id_map[HOTEL_ID_COL].astype(str)
                    
                    # Tạo dataframe map giữa itemId và hotel_info
                    hotel_comments = pd.read_csv(Path("./data/hotel_comments.csv"))
                    user_col = [c for c in hotel_comments.columns if "Reviewer Name" in c][0]
                    hotel_col = [c for c in hotel_comments.columns if "Hotel ID" in c][0]
                    
                    item_id_map = pd.DataFrame({
                        'itemId': pd.factorize(hotel_comments[hotel_col])[0],
                        HOTEL_ID_COL: hotel_comments[hotel_col]
                    }).drop_duplicates()

                    out_df = out_df.merge(item_id_map, on="itemId", how="left")
                    out_df = out_df.merge(hotel_info[[HOTEL_ID_COL, HOTEL_NAME_COL]], on=HOTEL_ID_COL, how="left")
                    
                    st.success(f"Top 10 gợi ý cho User ID: {user_id}")
                    st.dataframe(out_df[[HOTEL_NAME_COL, 'score']], use_container_width=True)

                spark.stop()
            except Exception as e:
                st.error(f"Lỗi: Mô hình ALS chưa sẵn sàng hoặc có lỗi xảy ra. Vui lòng chạy notebook để tạo mô hình.")
                st.error(str(e))
                if 'spark' in locals():
                    spark.stop()