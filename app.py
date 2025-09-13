# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re

# ---------------- Cấu hình trang (Page Config) ----------------
st.set_page_config(
    page_title="Agoda Hotel Recommendation",
    page_icon="🏨",
    layout="wide"
)

# ---------------- Các hàm tải và xử lý dữ liệu (với cache) ----------------


# 🔧 Spark session (đặt ở đầu file, chỉ tạo 1 lần)
from pyspark.sql import SparkSession
import streamlit as st

# def get_spark():
#     if "spark" not in st.session_state:
#         st.session_state.spark = (
#             SparkSession.builder
#             .master("local[*]")
#             .appName("AgodaALSApp")
#             # ⬇️ bắt buộc: đọc local không kiểm checksum
#             .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
#             .config("spark.hadoop.fs.file.impl.disable.cache", "true")
#             .getOrCreate()
#         )
#         # ép lại ở HadoopConf (phòng trường hợp builder chưa áp)
#         st.session_state.spark.sparkContext._jsc.hadoopConfiguration().set(
#             "fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem"
#         )
#     return st.session_state.spark

# # Nút reset Spark (để đảm bảo config mới được dùng)
# if st.sidebar.button("🔄 Reset Spark"):
#     try:
#         st.session_state.spark.stop()
#     except Exception:
#         pass
#     st.session_state.pop("spark", None)
#     st.experimental_rerun()

# spark = get_spark()

# ==== Utils: chuẩn hóa/ánh xạ tên cột ====
def normalize_col(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')

def auto_rename_columns(df: pd.DataFrame, wanted: dict) -> pd.DataFrame:
    """
    wanted = {
      'ChuẩnMuốnCó': ['candidates...', '...']
    }
    -> Đổi tên các cột đang có về đúng khóa 'ChuẩnMuốnCó' nếu tìm thấy ứng viên.
    """
    norm_map = {normalize_col(c): c for c in df.columns}
    rename_dict = {}
    for target, cands in wanted.items():
        cands_norm = [normalize_col(x) for x in cands + [target]]
        for k in cands_norm:
            if k in norm_map:
                rename_dict[norm_map[k]] = target
                break
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df

# ---------------- Các hàm tải và xử lý dữ liệu (với cache) ----------------
@st.cache_data
def load_main_data():
    """Tải và chuẩn bị dữ liệu chính từ hotel_info.csv và hotel_comments.csv."""
    # Hotel info
    try:
        hotel_df = pd.read_csv("./data/hotel_info.csv", encoding="utf-8")
    except Exception:
        hotel_df = pd.read_csv("./data/hotel_info.csv", encoding="latin-1")
    hotel_df.columns = hotel_df.columns.str.strip()

    # Ánh xạ cột thường gặp -> tên chuẩn dùng trong app
    hotel_df = auto_rename_columns(
        hotel_df,
        {
            "Hotel_ID": ["hotel_id", "hotelid", "id_hotel", "property_id", "propertyid", "id"],
            "Hotel_Name": ["hotel_name", "name_hotel", "property_name", "name"],
            "Hotel_Address": ["hotel_address", "address", "addr"],
            "Hotel_Description": ["hotel_description", "description", "desc", "about", "overview", "summary"],
            "Image_URL": ["image_url", "image", "photo", "thumbnail"],
            "Hotel_Rank": ["hotel_rank", "rank", "stars", "rating_class", "star_rating"],
            "Total_Score": ["total_score", "score", "avg_score", "overall_score"]
        }
    )

    # Nếu thiếu mô tả thì ghép tạm từ tên + địa chỉ để TF-IDF vẫn chạy
    if "Hotel_Description" not in hotel_df.columns:
        cols = [c for c in ["Hotel_Name", "Hotel_Address"] if c in hotel_df.columns]
        if cols:
            hotel_df["Hotel_Description"] = hotel_df[cols].astype(str).fillna("").agg(" ".join, axis=1)
        else:
            hotel_df["Hotel_Description"] = ""

    if "Total_Score" not in hotel_df.columns:
        hotel_df["Total_Score"] = np.random.uniform(7.5, 9.8, size=len(hotel_df)).round(1)

    # Comments / ratings
    try:
        comments_df = pd.read_csv("./data/hotel_comments.csv", encoding="utf-8")
    except Exception:
        comments_df = pd.read_csv("./data/hotel_comments.csv", encoding="latin-1")

    comments_df.columns = comments_df.columns.str.strip()
    comments_df = auto_rename_columns(
        comments_df,
        {
            "Reviewer_Name": ["reviewer_name", "reviewer", "user", "user_name", "username", "customer_name", "author", "name"],
            "Hotel_ID": ["hotel_id", "hotelid", "id_hotel", "property_id", "propertyid", "id"]
        }
    )

    # Cảnh báo dễ hiểu nếu vẫn thiếu cột bắt buộc
    missing = [c for c in ["Reviewer_Name", "Hotel_ID"] if c not in comments_df.columns]
    if missing:
        st.error(
            "Thiếu cột bắt buộc trong file `hotel_comments.csv`: "
            + ", ".join(missing)
            + ".\nCác cột hiện có: "
            + ", ".join(list(comments_df.columns))
        )

    return hotel_df, comments_df

from pyspark.ml.recommendation import ALSModel
from pathlib import Path
import os, glob

def _delete_crc_under(dir_path: str):
    for fp in glob.glob(os.path.join(dir_path, "**", "*.crc"), recursive=True):
        try:
            os.remove(fp)
        except Exception:
            pass

@st.cache_resource
def load_als_model():
    # Tránh nhầm sang bản “- Copy”, và chuẩn hóa thành file:/// URI
    model_dir = Path("./outputs/models/best_als_model").resolve()
    st.info(f"Đang load ALS model từ: {model_dir}")  # để bạn nhìn đúng thư mục
    _delete_crc_under(str(model_dir))                 # ✅ bỏ CRC gây lệch checksum
    uri = model_dir.as_uri()                          # vd: file:///C:/.../best_als_model
    try:
        return ALSModel.load(uri)
    except Exception as e:
        # ép lại cấu hình lần nữa rồi thử lại
        spark.sparkContext._jsc.hadoopConfiguration().set(
            "fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem"
        )
        _delete_crc_under(str(model_dir))
        return ALSModel.load(uri)

@st.cache_resource
def create_tfidf_recommender(_hotel_df):
    """Tạo recommender dựa trên TF-IDF (chịu lỗi cột mô tả thiếu)."""
    desc_col = "Hotel_Description" if "Hotel_Description" in _hotel_df.columns else None
    if not desc_col:
        # fallback an toàn
        _hotel_df["__desc_fallback__"] = _hotel_df.astype(str).fillna("").agg(" ".join, axis=1)
        desc_col = "__desc_fallback__"
    corpus = _hotel_df[desc_col].astype(str).fillna("").apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

@st.cache_resource
def load_embeddings(model_name):
    """Tải các file embedding đã tính sẵn (Doc2Vec, SBERT)."""
    # Đổi 'output' -> 'outputs' cho đúng thư mục
    path = f"./outputs/models/{model_name}_emb.npy"
    try:
        embeddings = np.load(path)
        return embeddings
    except FileNotFoundError:
        st.warning(f"Không tìm thấy embeddings tại: {path}")
        return None


def preprocess_text(text):
    """Hàm tiền xử lý văn bản tiếng Việt."""
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_content_recommendations(hotel_index, sim_matrix, df, top_n=10):
    """Lấy gợi ý từ ma trận tương đồng đã có."""
    sim_scores = list(enumerate(sim_matrix[hotel_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    hotel_indices = [i[0] for i in sim_scores]
    return df.iloc[hotel_indices]

def display_recommendation_list(df_recommendations):
    """Hiển thị danh sách khách sạn được gợi ý một cách đẹp mắt."""
    if df_recommendations.empty:
        st.info("Không tìm thấy gợi ý nào.")
        return
    
    for index, row in df_recommendations.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(row.get('Image_URL', 'https://i.imgur.com/uR3sYyP.jpeg'))
            with col2:
                st.subheader(row['Hotel_Name'])
                st.caption(f"📍 {row.get('Hotel_Address', 'N/A')}")
                st.write(f"⭐ **Hạng:** {row.get('Hotel_Rank', 'N/A')} | 💯 **Điểm:** {row.get('Total_Score', 'N/A')}")
                
# ---------------- Chương trình chính ----------------

# --- Tải và chuẩn bị dữ liệu ---
hotel_df, comments_df = load_main_data()
tfidf_vectorizer, tfidf_matrix, tfidf_cosine_sim = create_tfidf_recommender(hotel_df)
d2v_embeddings = load_embeddings("d2v")
sbert_embeddings = load_embeddings("sbert")

# --- Giao diện ---
st.title("AGODA Hotel Recommendation")
st.caption("Get tailored hotel suggestions with advanced filtering and multiple recommendation models.")

with st.sidebar:
    st.image("logo.png", width=100)
    st.header("Suggestions")
    page = st.radio(
        "",
        ('by hotel description', 'by hotel name', 'by rating review (ALS)')
    )
    st.markdown("---")
    st.header("About this project")
    st.info("Đồ án tốt nghiệp ứng dụng các thuật toán gợi ý vào bài toán thực tế trên dữ liệu từ Agoda.")
    
# --- Hiển thị trang tương ứng ---
if page == 'by hotel description':
    st.header("Tìm kiếm theo mô tả")
    search_query = st.text_input("Ví dụ: khách sạn mát, rộng, gần biển, có trẻ em", placeholder="Nhập mô tả của bạn ở đây...")
    if search_query:
        with st.spinner("Đang tìm những khách sạn phù hợp nhất..."):
            query_vec = tfidf_vectorizer.transform([preprocess_text(search_query)])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = sim_scores.argsort()[-10:][::-1]
            recommendations = hotel_df.iloc[top_indices]
        display_recommendation_list(recommendations)

elif page == 'by hotel name':
    st.header("Tìm khách sạn tương tự")
    method = st.selectbox("Chọn mô hình gợi ý theo nội dung:", ["TF-IDF", "Doc2Vec", "SBERT"])
    selected_hotel_name = st.selectbox("Chọn một khách sạn bạn đã thích:", hotel_df['Hotel_Name'].unique())
    if selected_hotel_name:
        selected_hotel_index = hotel_df[hotel_df['Hotel_Name'] == selected_hotel_name].index[0]
        with st.spinner(f"Đang tìm các khách sạn tương tự bằng {method}..."):
            recommendations = pd.DataFrame() # Khởi tạo dataframe rỗng
            if method == "TF-IDF":
                recommendations = get_content_recommendations(selected_hotel_index, tfidf_cosine_sim, hotel_df)
            elif method == "Doc2Vec" and d2v_embeddings is not None:
                sim_matrix = cosine_similarity(d2v_embeddings)
                recommendations = get_content_recommendations(selected_hotel_index, sim_matrix, hotel_df)
            elif method == "SBERT" and sbert_embeddings is not None:
                sim_matrix = cosine_similarity(sbert_embeddings)
                recommendations = get_content_recommendations(selected_hotel_index, sim_matrix, hotel_df)
        st.subheader(f"Top 10 gợi ý tương tự '{selected_hotel_name}' theo {method}:")
        display_recommendation_list(recommendations)

elif page == 'by rating review (ALS)':
    st.header("Gợi ý cho khách hàng (Mô hình ALS)")
    st.info("Tính năng này sử dụng Spark và có thể mất một chút thời gian để khởi tạo lần đầu.")
    
    user_list = comments_df['Reviewer_Name'].unique()
    selected_user = st.selectbox("Chọn một khách hàng để xem gợi ý:", user_list)

    if st.button(f"Lấy gợi ý cho {selected_user}", type="primary"):
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

                # spark = SparkSession.builder.appName("ALSInference").master("local[*]").getOrCreate()
                
                user_map = pd.DataFrame({'Reviewer_Name': comments_df['Reviewer_Name'], 'userId': pd.factorize(comments_df['Reviewer_Name'])[0]}).drop_duplicates()
                selected_user_id = user_map[user_map['Reviewer_Name'] == selected_user]['userId'].iloc[0]
                
                model = ALSModel.load("./outputs/models/best_als_model")
                user_df = spark.createDataFrame([(selected_user_id,)], ["userId"])
                recs_spark = model.recommendForUserSubset(user_df, 10).first()

                if recs_spark and recs_spark['recommendations']:
                    recs_list = [(row['itemId'], row['rating']) for row in recs_spark['recommendations']]
                    recs_df = pd.DataFrame(recs_list, columns=['itemId', 'Score'])
                    
                    item_map = pd.DataFrame({'itemId': pd.factorize(comments_df['Hotel_ID'])[0], 'Hotel_ID': comments_df['Hotel_ID']}).drop_duplicates()
                    recs_df = recs_df.merge(item_map, on='itemId')
                    recs_df = recs_df.merge(hotel_df, on='Hotel_ID')
                    
                    st.subheader(f"Top 10 gợi ý cho khách hàng '{selected_user}':")
                    display_recommendation_list(recs_df)
                else:
                    st.info(f"Không có gợi ý nào cho khách hàng {selected_user}.")
                
                spark.stop()

            except Exception as e:
                st.error("Có lỗi xảy ra khi chạy mô hình ALS.")
                st.error(f"Chi tiết lỗi: {e}")
                if 'spark' in locals() and spark.getActiveSession():
                    spark.stop()
