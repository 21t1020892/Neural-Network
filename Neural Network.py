import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
import random
from datetime import datetime
import matplotlib.pyplot as plt
import traceback

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/21t1020892/PCA-t-SNE.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "21t1020892"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "xN8@Q7V@Pbr6CYZ"
    mlflow.set_experiment("PCA & t-SNE")
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI

# Hàm tải dữ liệu từ OpenML
@st.cache_data
def load_mnist_data():
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return X, y
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu MNIST từ OpenML: {str(e)}")
        return None, None

# Tab hiển thị dữ liệu
def data():
    st.header("📘 Dữ Liệu MNIST từ OpenML")
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.X = None
        st.session_state.y = None

    if st.button("⬇️ Tải dữ liệu từ OpenML"):
        with st.spinner("⏳ Đang tải dữ liệu MNIST từ OpenML..."):
            X, y = load_mnist_data()
            if X is not None and y is not None:
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success("✅ Dữ liệu đã được tải thành công!")
            else:
                st.error("❌ Không thể tải dữ liệu!")

    if st.session_state.data_loaded:
        X, y = st.session_state.X, st.session_state.y
        st.write(f"""
            **Thông tin tập dữ liệu MNIST:**
            - Tổng số mẫu: {X.shape[0]}
            - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
            - Số lớp: 10 (chữ số từ 0-9)
        """)

        st.subheader("Một số hình ảnh mẫu")
        n_samples = 10
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[row, col].set_title(f"Label: {y[idx]}")
            axes[row, col].axis("off")
        plt.tight_layout()
        st.pyplot(fig)

# Tab giải thích Neural Network (dựa trên bài viết)
# ... (Các import và hàm khác giữ nguyên)

def explain_nn():
    st.header("🧠 Neural Network - Mạng Nơ-ron Nhân tạo")

    st.subheader("1. Neural Network là gì?")
    st.markdown("""
    **Neural Network (Mạng Nơ-ron Nhân tạo)** là một mô hình học máy được thiết kế để mô phỏng cách hoạt động của hệ thần kinh trong não người. Nó bao gồm các **nơ-ron nhân tạo** được tổ chức thành nhiều lớp, kết nối với nhau qua các **trọng số (weights)**. Mục tiêu là học từ dữ liệu để dự đoán đầu ra dựa trên đầu vào, chẳng hạn như phân loại chữ số trong tập dữ liệu MNIST.
    """)

    st.subheader("2. Hoạt động của các nơ-ron")
    st.markdown("""
    Mỗi **nơ-ron** trong mạng nơ-ron nhận đầu vào từ các nơ-ron khác hoặc trực tiếp từ dữ liệu, sau đó xử lý thông tin qua một **hàm kích hoạt (activation function)** như sigmoid để tạo ra đầu ra. Quá trình này bao gồm:
    - **Tổng trọng số**: Kết hợp tuyến tính các đầu vào với trọng số và thêm độ lệch (bias):  
      $$ z = W \\cdot X + b $$
    - **Hàm kích hoạt**: Biến đổi z để đưa ra giá trị phi tuyến, ví dụ:  
      $$ a = \\text{sigmoid}(z) = \\frac{1}{1 + e^{-z}} $$
    """)
    st.image("https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/human_neuron_anatomy.png?w=717&ssl=1", 
             caption="Hoạt động của một nơ-ron trong mạng (Nguồn: nttuan8.com)", 
             use_column_width=True)

    st.subheader("3. Mô hình Neural Network")
    st.markdown("""
    Một mô hình neural network cơ bản bao gồm:
    - **Tầng đầu vào (Input Layer)**: Chứa dữ liệu đầu vào .
    - **Tầng ẩn (Hidden Layers)**: Xử lý dữ liệu qua các nơ-ron với hàm kích hoạt.
    - **Tầng đầu ra (Output Layer)**: Đưa ra kết quả dự đoán .
    """)
    st.image("https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/nn-1.png?resize=768%2C631&ssl=1", 
             caption="Cấu trúc mô hình neural network (Nguồn: nttuan8.com)", 
             use_column_width=True)

    st.subheader("4. Logistic Regression")
    st.markdown("""
    **Logistic Regression** là một trường hợp đặc biệt của neural network với chỉ một tầng và hàm kích hoạt sigmoid. Nó được dùng để phân loại nhị phân (0 hoặc 1), nhưng có thể mở rộng cho phân loại đa lớp bằng cách sử dụng softmax thay vì sigmoid. Công thức cơ bản:
    - Đầu ra:  
      $$ P(y=1|X) = \\text{sigmoid}(W \\cdot X + b) $$
    """)

    st.subheader("5. Mô hình tổng quát")
    st.markdown("""
    Mô hình neural network tổng quát hóa logistic regression bằng cách thêm nhiều tầng ẩn. Mỗi tầng ẩn học các đặc trưng phức tạp hơn từ dữ liệu:
    - Tầng 1: Học các đặc trưng cơ bản .
    - Tầng sâu hơn: Học các đặc trưng trừu tượng .
    Quá trình học dựa trên việc điều chỉnh trọng số để giảm thiểu hàm mất mát (loss function).
    """)

    st.subheader("6. Kí hiệu")
    st.markdown("""
    Các kí hiệu cơ bản trong neural network:
    - $X$: Vector đầu vào .
    - $W$: Ma trận trọng số (weights), ví dụ $W^{[1]}$ cho tầng 1.
    - $b$: Vector độ lệch (bias).
    - $z$: Tổng trọng số, $z = W \\cdot X + b$.
    - $a$: Đầu ra sau hàm kích hoạt, $a = \\text{sigmoid}(z)$.
    - $y$: Nhãn thực tế.
    - $\\hat{y}$: Nhãn dự đoán.
    - $\\eta$: Tốc độ học (learning rate).
    """)

    st.subheader("7. Feedforward (Lan truyền xuôi)")
    st.markdown("""
    **Feedforward** là quá trình truyền dữ liệu từ tầng đầu vào qua các tầng ẩn đến tầng đầu ra:
    1. Tính $z^{[l]} = W^{[l]} \\cdot a^{[l-1]} + b^{[l]}$ cho mỗi tầng $l$.
    2. Áp dụng hàm kích hoạt: $a^{[l]} = \\text{sigmoid}(z^{[l]})$.
    3. Lặp lại đến tầng đầu ra để có dự đoán $\\hat{y}$.
    """)
    st.image("https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/fw.png?w=1065&ssl=1", 
             caption="Quá trình feedforward trong mạng nơ-ron (Nguồn: nttuan8.com)", 
             use_column_width=True)


# ... (Các hàm khác như data(), split_data(), train(), du_doan(), show_experiment_selector(), main() giữ nguyên)

# Tab chia dữ liệu
def split_data():
    st.header("📌 Chia dữ liệu Train/Validation/Test")
    if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
        st.warning("⚠ Vui lòng tải dữ liệu từ tab 'Dữ Liệu' trước khi tiếp tục!")
        return

    X, y = st.session_state.X, st.session_state.y
    total_samples = X.shape[0]
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False

    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        try:
            indices = np.random.choice(total_samples, num_samples, replace=False)
            X_selected = X[indices]
            y_selected = y[indices]

            stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
            )

            stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size/(100 - test_size),
                stratify=stratify_option, random_state=42
            )

            st.session_state.total_samples = num_samples
            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test
            st.session_state.test_size = X_test.shape[0]
            st.session_state.val_size = X_val.shape[0]
            st.session_state.train_size = X_train.shape[0]
            st.session_state.data_split_done = True

            summary_df = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
            })
            st.success("✅ Dữ liệu đã được chia thành công!")
            st.table(summary_df)
        except Exception as e:
            st.error(f"❌ Lỗi khi chia dữ liệu: {str(e)}")
            traceback.print_exc()

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")

# Tab huấn luyện Neural Network
def train():
    st.header("⚙️ Huấn luyện Neural Network")
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train = st.session_state.X_train
    X_val = st.session_state.X_val
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_val = st.session_state.y_val
    y_test = st.session_state.y_test

    st.markdown("""
    ### 🧠 Neural Network 
    - **Tham số quan trọng**:
      - `hidden_layer_sizes`: Số nơ-ron trong các tầng ẩn.
      - `max_iter`: Số lần lặp tối đa.
      - `learning_rate_init`: Tốc độ học ban đầu.
    """)

    hidden_size = st.slider("Số nơ-ron lớp ẩn:", 50, 200, 100, step=10)
    max_iter = st.slider("Số lần lặp tối đa:", 50, 200, 100, step=25)  # Giảm max để tối ưu
    
    # Khởi tạo giá trị mặc định trong session_state nếu chưa có
    if "learning_rate_init" not in st.session_state:
        st.session_state.learning_rate_init = 0.001

    # Sử dụng st.number_input với key và format
    learning_rate_init = st.number_input(
        "Tốc độ học ban đầu:", 
        min_value=0.001, 
        max_value=0.7, 
        value=st.session_state.learning_rate_init,
        step=0.0001, 
        format="%.4f",
        key="learning_rate_input"
    )
   
    st.session_state.learning_rate_init = learning_rate_init

    n_folds = 3  # Cố định 3 folds để giảm tải (ẩn slider)
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_NN_Run")
    st.session_state["run_name"] = run_name if run_name else "Default_NN_Run"

    if "training_results" not in st.session_state:
        st.session_state.training_results = None

    if st.button("Huấn luyện mô hình"):
        # Kiểm tra xem có chạy trên Streamlit Cloud không
        is_cloud = os.getenv("STREAMLIT_CLOUD", False)
        if is_cloud:
            st.warning("⚠️ Chạy trên Streamlit Cloud có thể chậm do hạn chế tài nguyên.")

        with st.spinner("⏳ Đang khởi tạo huấn luyện..."):
            # Khởi tạo MLflow (nếu không chạy trên Cloud thì log, nếu Cloud thì bỏ qua artifact)
            if not is_cloud:
                with mlflow.start_run(run_name=f"NN_{st.session_state['run_name']}") as run:
                    st.write(f"Debug: Run Name trong MLflow: {run.info.run_name}")
                    mlflow.log_param("total_samples", st.session_state.total_samples)
                    mlflow.log_param("test_size", st.session_state.test_size)
                    mlflow.log_param("validation_size", st.session_state.val_size)
                    mlflow.log_param("train_size", st.session_state.train_size)
                    mlflow.log_param("hidden_layer_sizes", hidden_size)
                    mlflow.log_param("max_iter", max_iter)
                    mlflow.log_param("learning_rate_init", learning_rate_init)
                    mlflow.log_param("n_folds", n_folds)

            # Thiết lập progress bar
            progress_bar = st.progress(0)
            total_steps = n_folds + 1  # n_folds cho Cross-Validation + 1 cho fit cuối
            step_progress = 1.0 / total_steps

            try:
                model = MLPClassifier(
                    hidden_layer_sizes=(hidden_size,),
                    max_iter=max_iter,
                    learning_rate_init=learning_rate_init,
                    random_state=42
                )

                # Cross-Validation với tiến trình chi tiết
                st.write(f"🔍 Đánh giá mô hình qua Cross-Validation ({n_folds} folds)...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
                fold_results = []
                for i in range(n_folds):
                    current_progress = (i + 1) * step_progress
                    progress_bar.progress(current_progress)
                    with st.spinner(f"Đang xử lý Fold {i + 1}/{n_folds} ({current_progress * 100:.1f}%)..."):
                        fold_result = f"📌 Fold {i + 1} - Accuracy: {cv_scores[i]:.4f}"
                        st.write(fold_result)
                        fold_results.append(fold_result)
                        if not is_cloud:
                            mlflow.log_metric(f"accuracy_fold_{i+1}", cv_scores[i])

                mean_cv_score = cv_scores.mean()
                std_cv_score = cv_scores.std()

                # Huấn luyện cuối cùng
                with st.spinner(f"Đang huấn luyện mô hình cuối cùng ({(n_folds * step_progress * 100):.1f}% - 100%)..."):
                    model.fit(X_train, y_train)
                    progress_bar.progress(1.0)
                    y_pred = model.predict(X_test)
                    test_accuracy = accuracy_score(y_test, y_pred)

                # Log kết quả vào MLflow nếu không chạy trên Cloud
                if not is_cloud:
                    mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
                    mlflow.log_metric("cv_accuracy_std", std_cv_score)
                    mlflow.log_metric("test_accuracy", test_accuracy)
                    mlflow.sklearn.log_model(model, "neural_network")

                st.session_state.training_results = {
                    "cv_scores": fold_results,
                    "cv_accuracy_mean": mean_cv_score,
                    "cv_accuracy_std": std_cv_score,
                    "test_accuracy": test_accuracy,
                    "run_name": f"NN_{st.session_state['run_name']}",
                    "status": "success"
                }
                st.session_state['model'] = model

            except Exception as e:
                error_message = str(e)
                if not is_cloud:
                    mlflow.log_param("status", "failed")
                    mlflow.log_metric("cv_accuracy_mean", -1)
                    mlflow.log_metric("cv_accuracy_std", -1)
                    mlflow.log_metric("test_accuracy", -1)
                    mlflow.log_param("error_message", error_message)
                
                st.session_state.training_results = {
                    "error_message": error_message,
                    "run_name": f"NN_{st.session_state['run_name']}",
                    "status": "failed"
                }

    # Hiển thị kết quả sau huấn luyện
    if st.session_state.training_results:
        if st.session_state.training_results["status"] == "success":
            st.subheader("📊 Kết quả huấn luyện")
            st.write("🔍 **Đánh giá mô hình qua Cross-Validation:**")
            for fold_result in st.session_state.training_results["cv_scores"]:
                st.write(fold_result)
            st.success(f"📊 Cross-Validation Accuracy trung bình: {st.session_state.training_results['cv_accuracy_mean']:.4f} (±{st.session_state.training_results['cv_accuracy_std']:.4f})")
            st.success(f"✅ Độ chính xác trên test set: {st.session_state.training_results['test_accuracy']:.4f}")
            st.success(f"✅ Huấn luyện hoàn tất cho **{st.session_state.training_results['run_name']}**!")
            if not os.getenv("STREAMLIT_CLOUD"):
                st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
        else:
            st.error(f"❌ Lỗi khi huấn luyện mô hình: {st.session_state.training_results['error_message']}")
            if not os.getenv("STREAMLIT_CLOUD"):
                st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
# Tab dự đoán
def du_doan():
    st.header("✍️ Dự đoán số viết tay")

    # Kiểm tra MLflow đã khởi tạo chưa
    if 'mlflow_url' not in st.session_state:
        st.warning("⚠️ MLflow chưa được khởi tạo. Đang khởi tạo...")
        mlflow_input()

    # Lấy danh sách các run từ MLflow
    try:
        experiment_name = "MNIST_NeuralNetwork"
        experiments = mlflow.search_experiments()
        experiment = next((exp for exp in experiments if exp.name == experiment_name), None)
        if not experiment:
            st.error("❌ Không tìm thấy experiment 'MNIST_NeuralNetwork'!")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        successful_runs = runs[runs["status"] == "FINISHED"]  # Chỉ lấy các run thành công
        if successful_runs.empty:
            st.error("⚠️ Chưa có mô hình nào được huấn luyện thành công!")
            return

        run_options = {
            f"{row['tags.mlflow.runName']} (Run ID: {row['run_id'][:8]})": row["run_id"]
            for _, row in successful_runs.iterrows()
        }
        selected_run_name = st.selectbox("📌 Chọn mô hình đã huấn luyện:", list(run_options.keys()))
        selected_run_id = run_options[selected_run_name]

        # Tải mô hình từ MLflow
        model_uri = f"runs:/{selected_run_id}/neural_network"
        model = mlflow.sklearn.load_model(model_uri)
        st.success(f"✅ Đã chọn mô hình: {selected_run_name}")

    except Exception as e:
        st.error(f"❌ Lỗi khi truy cập MLflow hoặc tải mô hình: {str(e)}")
        traceback.print_exc()
        return

    # Chọn phương thức nhập liệu
    input_method = st.radio("📥 Chọn phương thức nhập liệu:", ("Vẽ tay", "Tải ảnh lên"))

    img = None
    if input_method == "Vẽ tay":
        if "key_value" not in st.session_state:
            st.session_state.key_value = str(random.randint(0, 1000000))

        if st.button("🔄 Tải lại nếu không thấy canvas"):
            st.session_state.key_value = str(random.randint(0, 1000000))

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key=st.session_state.key_value,
            update_streamlit=True
        )
        if st.button("Dự đoán số từ bản vẽ"):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
                img = img.resize((28, 28)).convert("L")
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.reshape(1, -1)
            else:
                st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

    else:
        uploaded_file = st.file_uploader("📤 Tải ảnh lên (định dạng PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh đã tải lên", width=150)
            if st.button("Dự đoán số từ ảnh"):
                img = Image.open(uploaded_file).convert("L")
                img = img.resize((28, 28))
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.reshape(1, -1)

    # Dự đoán và hiển thị kết quả
    if img is not None:
        st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
        prediction = model.predict(img)
        st.subheader(f"🔢 Dự đoán: {prediction[0]}")

        # Tính độ tin cậy
        confidence_scores = model.predict_proba(img)[0]  # Xác suất cho từng lớp (0-9)
        # Chú thích: Độ tin cậy (confidence) là xác suất cao nhất của lớp được dự đoán
        # confidence_scores là một mảng 10 phần tử (mỗi phần tử là xác suất của một lớp từ 0-9)
        # predicted_class_confidence là xác suất của lớp được dự đoán (prediction[0])
        predicted_class_confidence = confidence_scores[int(prediction[0])]  # Lấy xác suất của lớp dự đoán
        
        # Hiển thị độ tin cậy với định dạng rõ ràng và không làm tròn quá mức
        st.write(f"📈 **Độ tin cậy:** {predicted_class_confidence:.4f} ({predicted_class_confidence * 100:.2f}%)")
        #st.markdown("""
        #*Chú thích*: Độ tin cậy được tính bằng phương thức `predict_proba` của mô hình, trả về xác suất dự đoán cho từng lớp (0-9). Giá trị hiển thị là xác suất của lớp được dự đoán, nằm trong khoảng [0, 1], với tổng xác suất của tất cả các lớp bằng 1.
        #""")

        # Hiển thị xác suất cho từng lớp
        st.write("**Xác suất cho từng lớp (0-9):**")
        confidence_df = pd.DataFrame({"Nhãn": range(10), "Xác suất": confidence_scores})
        st.bar_chart(confidence_df.set_index("Nhãn"))

        # Hiển thị thông tin chi tiết từ MLflow
        st.subheader("📊 Thông tin chi tiết từ MLflow")
        show_experiment_selector()

# (Hàm show_experiment_selector() giữ nguyên như trước)

# Tab MLflow Experiments
def show_experiment_selector():
    if 'mlflow_url' not in st.session_state:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")
        mlflow_input()

    st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    experiment_name = "MNIST_NeuralNetwork"
    
    try:
        experiments = mlflow.search_experiments()
        selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

        if not selected_experiment:
            st.error(f"❌ Không tìm thấy Experiment '{experiment_name}'!", icon="🚫")
            return

        st.subheader(f"📌 Experiment: {experiment_name}")
        st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
        st.write(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
        st.write(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

        runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])
        if runs.empty:
            st.warning("⚠ Không có runs nào trong experiment này!", icon="🚨")
            return

        st.subheader("🏃‍♂️ Danh sách Runs")
        run_info = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
            run_info.append((run_name, run_id))

        run_name_to_id = dict(run_info)
        run_names = list(run_name_to_id.keys())

        selected_run_name = st.selectbox("🔍 Chọn Run để xem chi tiết:", run_names, key="run_selector_du_doan")
        selected_run_id = run_name_to_id[selected_run_name]
        selected_run = mlflow.get_run(selected_run_id)

        if selected_run:
            st.markdown(f"<h3 style='color: #28B463;'>📌 Chi tiết Run: {selected_run_name}</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("#### ℹ️ Thông tin cơ bản")
                st.info(f"**Run Name:** {selected_run_name}")
                st.info(f"**Run ID:** `{selected_run_id}`")
                st.info(f"**Trạng thái:** {selected_run.info.status}")
                start_time_ms = selected_run.info.start_time
                start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
                st.info(f"**Thời gian chạy:** {start_time}")

            with col2:
                params = selected_run.data.params
                if params:
                    st.write("#### ⚙️ Parameters")
                    with st.container(height=200):
                        st.json(params)

                metrics = selected_run.data.metrics
                if metrics:
                    st.write("#### 📊 Metrics")
                    with st.container(height=200):
                        st.json(metrics)
    except Exception as e:
        st.error(f"❌ Lỗi khi truy cập MLflow: {str(e)}")
        traceback.print_exc()
# Giao diện chính
def main():
    if "mlflow_initialized" not in st.session_state:
        mlflow_input()
        st.session_state.mlflow_initialized = True

    st.title("🖊️ MNIST Neural Network Classification App")
    tabs = st.tabs(["📘 Dữ Liệu", "🧠 Neural Network", "📌 Chia Dữ Liệu", "⚙️ Huấn Luyện", "🔢 Dự Đoán"])

    with tabs[0]:
        data()
    with tabs[1]:
        explain_nn()
    with tabs[2]:
        split_data()
    with tabs[3]:
        train()
    with tabs[4]:
        du_doan()

if __name__ == "__main__":
    main()