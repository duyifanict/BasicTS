import os
from typing import Optional
import streamlit as st
import pandas as pd
from engine.utils import get_baseline_config_dict, get_ckpt_config_dict, load_dataframe
from engine.engine import inference_engine

@st.cache_resource
def load_input_data(data_file):
    result = []
    for line in data_file.readlines():
        result.append(line.decode("utf-8").strip().split(","))
    return result

@st.cache_resource
def load_model(cfg:str, ckpt:str, device:str, gpu:Optional[str]):
    return inference_engine(cfg, ckpt, device, gpu)

load_model_container = st.container(border=True)
with load_model_container:
    st.write("load model")
    baseline_configs = get_baseline_config_dict()
    config_dir = st.selectbox("config dir", list(baseline_configs.keys()))
    config_file = st.selectbox("config file", baseline_configs[config_dir])

    ckpt_configs = get_ckpt_config_dict()
    ckpt_cfg_list = [x for x in list(ckpt_configs.keys()) if config_dir in x]
    if len(ckpt_cfg_list) == 0:
        st.write("no checkpoint found for this config, please train first")
    else:
        ckpt_dir = st.selectbox("checkpoint dir", ckpt_cfg_list)
        ckpt_file = st.selectbox("checkpoint file", ckpt_configs[ckpt_dir], index=len(ckpt_configs[ckpt_dir])-1)

    device_type = st.selectbox("device type", ["cpu", "gpu", "mlu"])
    gpus = None
    if device_type == "gpu":
        gpus = st.text_input("gpus", "0")

    if len(ckpt_cfg_list) != 0:
        submitted = st.button("load model")
        if submitted:
            cfg_path = os.path.join("baselines", config_dir, config_file)
            ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", ckpt_dir, ckpt_file)
            st.write(cfg_path)
            st.session_state["model"] = load_model(cfg_path, ckpt_path, device_type, gpus)
            st.write("model loaded")

load_data_container = st.container(border=True)
with load_data_container:
    st.write("load input data")
    input_data_file = st.file_uploader("input data file", type=["csv"])
    show_data_df = None
    if input_data_file:
        st.session_state["input_data_list"] = load_input_data(input_data_file)

        st.write("show input data (last 50 rows most)")
        if len(st.session_state["input_data_list"]) > 50:
            show_data = st.session_state["input_data_list"][-50:]
        else:
            show_data = st.session_state["input_data_list"]

        show_data_df = load_dataframe(show_data)
        st.write(show_data_df)
        st.write("data loaded")

    if st.checkbox("show data plot (first 10 columns most)"):
        if show_data_df is not None:
            st.line_chart(show_data_df.iloc[:, :10])
        else:
            st.write("no input data loaded")

inference_container = st.container(border=True)
with inference_container:
    st.write("inference")

    # check model loaded
    if "model" not in st.session_state:
        st.write("model not loaded")
    else:
        if st.button("inference"):
            st.session_state["prediction"] = st.session_state["model"].inference(st.session_state["input_data_list"])
        if "prediction" in st.session_state:
            st.write("inference executed")
            st.write("show prediction data (last 50 rows most)")
            show_pred_data, datetime_data = st.session_state["prediction"]
            if len(show_pred_data) > 50:
                show_pred_data = show_pred_data[-50:]

            show_pred_data_df = pd.DataFrame(show_pred_data)
            show_pred_data_df.index = datetime_data
            st.write(show_pred_data_df)

            if st.checkbox("show prediction plot (first 10 columns most)"):
                if show_pred_data_df is not None:
                    st.line_chart(show_pred_data_df.iloc[:, :10])
                else:
                    st.write("no prediction data")

            pred_data, datetime_data = st.session_state["prediction"]
            output_pd = pd.DataFrame(pred_data)
            output_pd.index = datetime_data
            if st.download_button("Download CSV", output_pd.to_csv().encode("utf-8"), file_name="prediction.csv", mime="text/csv"):
                st.write("download success")


