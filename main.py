
import data_handling.json_handler as jh
import implementation.experiment as ex
import implementation.plotter as pl
import data_handling.ctr_handler as ch

def main():
    #ch.transform_ctr_seq_time_100k()
    #ex.run_show_server_occ()
    #ex.run_single_experiment()
    #ch.save_ctr_seq_time_100k()
    #ex.run_bounded_server_occ()
    #pl.plot_occ()
    #pl.plot_infin_cap_mtf()
    #pl.plot_occ_inf_cap()
    #ex.run_paper_experiment_1_unbound()
    #ex.run_paper_experiment_1()
    #pl.sum_results_all_ps()
    #pl.plot_3d_all_ps_fixed_server_n()
    #pl.line_plot_freq()
    #pl.plot_occ(cap="4")

    #ex.run_experiment_temp()
    #ex.run_experiment_ctr()

    #jh.generate_serv_ins_del_temp()
    #jh.generate_serv_ins_del_ctr()
    #jh.get_serv_ins_del_temp()
    #ex.run_show_server_access_ctr()

    # Fig.8
    #pl.plot_2d_all_ps_fixed_sever_n()
    pl.plot_2d_all_ps_fixed_sever_n_more_sim()

    # Fig. 3
    #ex.run_show_server_access_ctr()
    #pl.plot_server_acc(limit=1000)     # plot only. Maintains data, limit got to be same as experiment run

    # Fig. 2+7
    #ex.run_server_occ(infin=True)      # AND mult=True OR no parameter
    #pl.plotting_occ_all()

    # Fig. 9
    #pl.line_plot_n_servers()
    #pl.line_plot_stale()

if __name__ == '__main__':
    main()

