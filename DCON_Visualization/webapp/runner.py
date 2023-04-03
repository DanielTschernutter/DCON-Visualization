import streamlit as st
import pandas as pd
import io

from PIL import Image
from importlib import resources
from config import Config
from webapp.views import dataset_navigation_bar
from webapp.text import (
    get_DCON_markdown,
    get_Adam_markdown,
    get_Dataset_markdown
)
from webapp.utils import (
    create_data_folder,
    check_loaded_datasets
)
from webapp.loader import load_dataset_from_UCL
from webapp.benchmarking import (
    train_and_evaluate_DCON,
    train_and_evaluate_Keras
)
from webapp.plots import (
    create_line_plot_loss,
    create_line_plot_norm_diffs,
    create_bar_plot_train_loss,
    create_bar_plot_test_loss,
    create_bar_plot_cputime
)

def webapp(config: Config):

    create_data_folder(config)
    loaded_datasets = check_loaded_datasets(config)

    selected_dataset_ind, options = dataset_navigation_bar(config)

    with resources.open_binary('DCON_Visualization', 'DCON_logo.png') as fp:
        img = fp.read()
    image = Image.open(io.BytesIO(img))

    with st.columns(3)[1]:
        placeholder = st.empty()
        placeholder_image = st.empty()
        placeholder_image.image(image,use_column_width=True)
        clicked = placeholder.button("Perform experiments", type='primary', use_container_width=True)
    
    placeholder_expander = st.empty()
    
    with placeholder_expander.expander("Details"):
        placeholder_title = st.empty()
        placeholder_title.markdown("<h1 style='text-align: center; color: black;'>DCON vs. Adam</h1>", unsafe_allow_html=True)

        placeholder_DCON = st.empty()
        markdown_text = get_DCON_markdown()
        placeholder_DCON.markdown(markdown_text)

        placeholder_Adam = st.empty()
        markdown_text = get_Adam_markdown()
        placeholder_Adam.markdown(markdown_text)

        placeholder_Dataset = st.empty()
        markdown_text = get_Dataset_markdown()
        placeholder_Dataset.markdown(markdown_text)

    status = True

    if clicked:

        placeholder.empty()
        placeholder_title.empty()
        placeholder_DCON.empty()
        placeholder_Adam.empty()
        placeholder_Dataset.empty()
        placeholder_image.empty()
        placeholder_expander.empty()
        
        with st.columns(3)[1]:
            with st.spinner("Performing experiments..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(0,"Loading dataset")
                if not loaded_datasets[selected_dataset_ind]:
                    status = load_dataset_from_UCL(config,selected_dataset_ind)
                    if not status:
                        st.error('Error while loading dataset.', icon="🚨")
                    loaded_datasets[selected_dataset_ind] = True

                if status:
                    progress_bar.progress(25,"Training DCON")
                    results = train_and_evaluate_DCON(config,selected_dataset_ind,options)
                    
                    progress_bar.progress(50,"Training Adam")
                    results.update(
                        train_and_evaluate_Keras(config,selected_dataset_ind,options)
                    )
                    
                    progress_bar.progress(75,"Generating plots")
                    line_plot_loss = create_line_plot_loss(results,options)
                    line_plot_norm_diffs = create_line_plot_norm_diffs(results)
                    bar_plot_train_loss = create_bar_plot_train_loss(results)
                    bar_plot_test_loss = create_bar_plot_test_loss(results)
                    bar_plot_cputime = create_bar_plot_cputime(results)
                    
                    progress_bar.progress(100)
                    progress_bar.empty()

        if status:
            st.title("Results for "+config.DATASET_NAMES[selected_dataset_ind])
            st.header("DCON vs. Adam")
            col1, col2 = st.columns(2)
            
            mse_train = results['train_loss_DCON']
            mse_train_imp = 100*((results['train_loss_DCON']/results['train_loss_Keras'])-1.0)
            col1.metric("MSE Training", "{:.4f}".format(mse_train), "{:.2f}%".format(mse_train_imp), delta_color="inverse")
            
            mse_test = results['test_loss_DCON']
            mse_test_imp = 100*((results['test_loss_DCON']/results['test_loss_Keras'])-1.0)
            col2.metric("MSE Test", "{:.4f}".format(mse_test), "{:.2f}%".format(mse_test_imp), delta_color="inverse")
            
            with st.expander("Details"):
                tab1, tab2, tab3, tab4 = st.tabs(["DCON vs. Adam (Train MSE)",
                                                  "DCON vs. Adam (Test MSE)",
                                                  "DCON vs. Adam (CPU Time)",
                                                  "DCON Details"])    
            
                with tab1:
                    st.plotly_chart(bar_plot_train_loss, theme="streamlit", use_container_width=True)
                with tab2:
                    st.plotly_chart(bar_plot_test_loss, theme="streamlit", use_container_width=True)
                with tab3:
                    st.plotly_chart(bar_plot_cputime, theme="streamlit", use_container_width=True)
                with tab4:
                    st.plotly_chart(line_plot_loss, theme="streamlit", use_container_width=True)
                    st.plotly_chart(line_plot_norm_diffs, theme="streamlit", use_container_width=True)
                    
            with st.expander("Tuned Hyperparameters Adam"):
                df = pd.DataFrame([[str(results['keras_hyperparameters']['lr']),
                                    str(results['keras_hyperparameters']['beta_1']),
                                    str(results['keras_hyperparameters']['reg_param']),
                                    str(results['keras_hyperparameters']['batch_size'])]],
                                    columns=["Learning Rate","Momentum","Regularization","Batch Size"])
                
                # CSS to inject contained in a string
                hide_table_row_index = """
                            <style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style>
                            """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                st.table(df)