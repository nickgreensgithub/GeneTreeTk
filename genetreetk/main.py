###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

import os
import sys
import logging

import dendropy

from biolib.common import check_file_exists, make_sure_path_exists
from biolib.external.execute import check_dependencies

from genetreetk.blast_workflow import BlastWorkflow
from genetreetk.concatenate import Concatenate
from genetreetk.reduce_workflow import Reduce
from genetreetk.bootstrap import Bootstrap
from genetreetk.prune import Prune
from genetreetk.prokka import Prokka
from genetreetk.create_database import CreateDatabase
from genetreetk.tree_compare import TreeCompare
from genetreetk.orthologue_workflow import OrthologueWorkflow
from genetreetk.arb_db_creator import ArbDbCreator
from genetreetk.main import OptionsParser

from biolib.misc.custom_help_formatter import CustomHelpFormatter
from biolib.logger import logger_setup
import argparse

def version():
    """Read program version from file."""
    import genetreetk
    version_file = open(os.path.join(genetreetk.__path__[0], 'VERSION'))
    return version_file.readline().strip()


def print_help():
    print('')
    print('                ...::: GeneTreeTk v' + version() + ' :::...''')
    print('''\

  Gene tree inference:
    blast      -> Infer gene tree using BLAST
    concat     -> Infer concatenated gene tree
    orthologue -> Infer gene trees after orthologue clustering

  Tree utilities:
    reduce    -> Infer tree for reduced set of genes
    bootstrap -> Calculate bootstrap support for tree
    prune     -> Prune tree to a specific set of extant taxa

  Database utilities:
    prokka    -> Run Prokka across multiple genome bins
    create_db -> Create dereplicated GeneTreeTk-compatible database
    arb_db    -> Create an ARB DB from a list of protein sequences

  Compare trees:
    robinson_foulds  -> Calculate Robinson-Foulds distance between trees
    supported_splits -> Supported bipartitions of common taxa shared between two trees
    missing_splits   -> Report supported bipartitions in reference tree not in comparison tree

  Reroot tree:
    midpoint -> Reroot tree at midpoint

  Use: genetreetk <command> -h for command specific help.

  Feature requests or bug reports can be sent to Donovan Parks (donovan.parks@gmail.com)
    or posted on GitHub (https://github.com/dparks1134/GeneTreeTk).
    ''')


def run_program():
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(help="--", dest='subparser_name')

    # options for inferring a gene tree using blast
    blast_parser = subparsers.add_parser('blast',
                                         add_help=False,
                                         formatter_class=CustomHelpFormatter,
                                         description='Infer gene tree using BLAST.')

    req_args = blast_parser.add_argument_group('required arguments')
    req_args.add_argument('-q', '--query_proteins', help='protein sequences for identifying homologs (fasta format)',
                          required=True)
    req_args.add_argument('-d', '--db_file', help='BLAST database of reference proteins', required=True)
    req_args.add_argument('-t', '--taxonomy_file', help='taxonomic assignment of each reference genomes', required=True)
    req_args.add_argument('-o', '--output_dir', help='output directory', required=True)

    optonal_args = blast_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--custom_db_file', help='BLAST database of additional proteins', default=None)
    optonal_args.add_argument('--custom_taxonomy_file', help='taxonomic assignment of each genomes in custom database',
                              default=None)

    optonal_args.add_argument('--homology_search', choices=['blastp-fast', 'blastp', 'diamond'],
                              help="type of homology search to perform", default='blastp-fast')
    optonal_args.add_argument('-e', '--evalue', help='evalue cutoff for identifying homologs', type=float, default=1e-5)
    optonal_args.add_argument('-p', '--per_identity', help="percent amino acid identity for identifying homologs",
                              type=float, default=30.0)
    optonal_args.add_argument('-a', '--per_aln_len',
                              help="percent alignment length of query sequence for identifying homologs", type=float,
                              default=50.0)
    optonal_args.add_argument('-m', '--max_matches', help="maximum number of matches per query protein", type=int,
                              default=50000)
    optonal_args.add_argument('--restrict_taxon',
                              help='restrict alignment to specific taxonomic group (e.g., d__Archaea)', default=None)

    optonal_args.add_argument('--msa_program', choices=['mafft', 'muscle'],
                              help="program to use for multiple sequence alignment (MSA)", default='mafft')

    optonal_args.add_argument('--min_per_taxa', help='minimum percentage of taxa required to retain column in MSA',
                              type=float, default=50.0)
    optonal_args.add_argument('--consensus',
                              help='minimum percentage of the same amino acid required to retain column in MSA',
                              type=float, default=0.0)
    optonal_args.add_argument('--min_per_bp',
                              help='minimum percentage of base pairs in MSA required to keep a trimmed sequence',
                              type=float, default=50.0)
    optonal_args.add_argument('--use_trimAl',
                              help='filter columns in MSA using trimAl (other trimming parameters are ignored)',
                              action='store_true')

    optonal_args.add_argument('--tree_program', choices=['fasttree', 'raxml'], help="program to use for tree inference",
                              default='fasttree')
    optonal_args.add_argument('--prot_model', choices=['LG', 'WAG', 'AUTO'],
                              help='protein substitution model for tree inference', default='LG')

    optonal_args.add_argument('--skip_rooting', help="do not perform midpoint rooting before decorating tree",
                              action='store_true')

    optonal_args.add_argument('--cpus', type=int, help='CPUs to use throughout the process', default=1)
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # infer concatenated gene tree
    concat_parser = subparsers.add_parser('concat',
                                          add_help=False,
                                          formatter_class=CustomHelpFormatter,
                                          description='Infer concatenated gene tree.')

    req_args = concat_parser.add_argument_group('required arguments')
    req_args.add_argument('-g', '--gene_dirs', nargs='+',
                          help='output directories for individual genes produced by GeneTreeTk', required=True)
    req_args.add_argument('-o', '--output_dir', help='output directory', required=True)

    optonal_args = concat_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--split_chars', nargs='?',
                              help='character(s) to use for splitting taxa and gene identifiers', default='~')

    optonal_args.add_argument('--min_per_gene', help='minimum percentage of genes required to retain taxa', type=float,
                              default=50.0)
    optonal_args.add_argument('--min_per_bps', help='minimum percentage of base pairs required to retain taxa',
                              type=float, default=50.0)

    optonal_args.add_argument('--tree_program', choices=['fasttree', 'raxml'], help="program to use for tree inference",
                              default='fasttree')
    optonal_args.add_argument('--prot_model', choices=['LG', 'WAG', 'AUTO'],
                              help='protein substitution model for tree inference', default='LG')

    optonal_args.add_argument('--cpus', type=int, help='CPUs to use throughout the process', default=1)
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # options for inferring a gene tree after orthologue clustering
    blast_parser = subparsers.add_parser('orthologue',
                                         add_help=False,
                                         formatter_class=CustomHelpFormatter,
                                         description='Infer gene tree using BLAST.')

    req_args = blast_parser.add_argument_group('required arguments')
    req_args.add_argument('-q', '--query_proteins', help='protein sequences for identifying homologs (fasta format)',
                          required=True)
    req_args.add_argument('-d', '--db_file', help='BLAST database of reference proteins', required=True)
    req_args.add_argument('-t', '--taxonomy_file', help='taxonomic assignment of each reference genomes', required=True)
    req_args.add_argument('-o', '--output_dir', help='output directory', required=True)

    optonal_args = blast_parser.add_argument_group('optional arguments')
    # optonal_args.add_argument('--custom_db_file', help='BLAST database of additional proteins', default=None)
    # optonal_args.add_argument('--custom_taxonomy_file', help='taxonomic assignment of each genomes in custom database', default=None)

    optonal_args.add_argument('--homology_search', choices=['blastp-fast', 'blastp', 'diamond'],
                              help="type of homology search to perform", default='blastp-fast')
    optonal_args.add_argument('-e', '--evalue', help='evalue cutoff for identifying homologs', type=float, default=1e-5)
    optonal_args.add_argument('-p', '--per_identity', help="percent amino acid identity for identifying homologs",
                              type=float, default=30.0)
    optonal_args.add_argument('-a', '--per_aln_len',
                              help="percent alignment length of query sequence for identifying homologs", type=float,
                              default=50.0)
    optonal_args.add_argument('-m', '--max_matches', help="maximum number of matches per query protein", type=int,
                              default=50000)
    optonal_args.add_argument('--restrict_taxon',
                              help='restrict alignment to specific taxonomic group (e.g., d__Archaea)', default=None)

    optonal_args.add_argument('--msa_program', choices=['mafft', 'muscle'],
                              help="program to use for multiple sequence alignment (MSA)", default='mafft')

    optonal_args.add_argument('--min_per_taxa', help='minimum percentage of taxa required to retain column in MSA',
                              type=float, default=50.0)
    optonal_args.add_argument('--consensus',
                              help='minimum percentage of the same amino acid required to retain column in MSA',
                              type=float, default=0.0)
    optonal_args.add_argument('--min_per_bp',
                              help='minimum percentage of base pairs in MSA required to keep a trimmed sequence',
                              type=float, default=50.0)
    optonal_args.add_argument('--use_trimAl',
                              help='filter columns in MSA using trimAl (other trimming parameters are ignored)',
                              action='store_true')

    optonal_args.add_argument('--tree_program', choices=['fasttree', 'raxml'], help="program to use for tree inference",
                              default='fasttree')
    optonal_args.add_argument('--prot_model', choices=['LG', 'WAG', 'AUTO'],
                              help='protein substitution model for tree inference', default='LG')

    optonal_args.add_argument('--skip_rooting', help="do not perform midpoint rooting before decorating tree",
                              action='store_true')

    optonal_args.add_argument('--cpus', type=int, help='CPUs to use throughout the process', default=1)
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # infer tree over a reduced set of genes
    reduce_parser = subparsers.add_parser('reduce',
                                          add_help=False,
                                          formatter_class=CustomHelpFormatter,
                                          description='Infer tree for reduced set of genes.')

    req_args = reduce_parser.add_argument_group('required arguments')
    req_args.add_argument('-i', '--homolog_file',
                          help='file containing unaligned homologs used to infer initial tree (fasta format)',
                          required=True)
    req_args.add_argument('-g', '--gene_ids', help='gene IDs to retain in reduced tree (one id per line)',
                          required=True)
    req_args.add_argument('-t', '--taxonomy_file', help='taxonomic assignment of genes in gene tree', required=True)
    req_args.add_argument('-o', '--output_dir', help='output directory', required=True)

    optonal_args = reduce_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--msa_program', choices=['mafft', 'muscle'],
                              help="program to use for multiple sequence alignment (MSA)", default='mafft')
    optonal_args.add_argument('--min_per_taxa', help='minimum percentage of taxa required to retain column in MSA',
                              type=float, default=50.0)
    optonal_args.add_argument('--consensus',
                              help='minimum percentage of the same amino acid required to retain column in MSA',
                              type=float, default=0.0)
    optonal_args.add_argument('--min_per_bp',
                              help='minimum percentage of base pairs in MSA required to keep a trimmed sequence',
                              type=float, default=50.0)
    optonal_args.add_argument('--use_trimAl',
                              help='filter columns in MSA using trimAl (other trimming parameters are ignored)',
                              action='store_true')
    optonal_args.add_argument('--tree_program', choices=['fasttree', 'raxml'], help="program to use for tree inference",
                              default='fasttree')
    optonal_args.add_argument('--prot_model', choices=['LG', 'WAG', 'AUTO'],
                              help='protein substitution model for tree inference', default='LG')
    optonal_args.add_argument('--cpus', type=int, help='CPUs to use throughout the process', default=1)
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # calculate bootstrap support
    bootstrap_parser = subparsers.add_parser('bootstrap',
                                             add_help=False,
                                             formatter_class=CustomHelpFormatter,
                                             description='Calculate bootstrap support for tree.')

    req_args = bootstrap_parser.add_argument_group('required arguments')
    req_args.add_argument('-t', '--tree', help='tree requiring bootstrap support values', required=True)
    req_args.add_argument('-m', '--msa_file', help='multiple sequence alignment used to infer tree (fasta format)',
                          required=True)
    req_args.add_argument('-o', '--output_dir', help='output directory', required=True)

    optonal_args = bootstrap_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('-r', '--num_replicates', help='number of bootstrap replicates to perform', type=int,
                              default=100)
    optonal_args.add_argument('--tree_program', choices=['fasttree', 'raxml'],
                              help="program to use for inferring bootstrap trees", default='fasttree')
    optonal_args.add_argument('--prot_model', choices=['LG', 'WAG'],
                              help='protein substitution model for inferring bootstrap trees', default='LG')
    optonal_args.add_argument('--cpus', type=int, help='CPUs to use throughout the process', default=1)
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # Prune command
    prune_parser = subparsers.add_parser('prune',
                                         add_help=False,
                                         formatter_class=CustomHelpFormatter,
                                         description='Prune tree to a specific set of extant taxa.')
    req_args = prune_parser.add_argument_group('required arguments')
    req_args.add_argument('-i', '--tree', help='input tree in Newick format')
    req_args.add_argument('-t', '--taxa_to_retain', help='input file specify taxa to retain')
    req_args.add_argument('-o', '--output_tree', help='pruned output tree')

    optonal_args = prune_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # run prokka across multiple bins
    run_prokka_parser = subparsers.add_parser('prokka',
                                              add_help=False,
                                              formatter_class=CustomHelpFormatter,
                                              description='Run Prokka across multiple genome bins.')

    req_args = run_prokka_parser.add_argument_group('required arguments')
    req_args.add_argument('-g', '--genome_dir', help="directory containing genome bins")
    req_args.add_argument('-o', '--output_dir', help='directory to store results')

    optonal_args = run_prokka_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('-x', '--extension', default='fna',
                              help="extension of genome FASTA files (other files in directory are ignored)")
    optonal_args.add_argument('--kingdom', help='kingdom to use for gene annotation', choices=['Archaea', 'Bacteria'],
                              default='Bacteria')
    optonal_args.add_argument('--cpus', type=int, default=1, help='number of cpus to use')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # create database that is compatible with mingle
    create_db_parser = subparsers.add_parser('create_db',
                                             add_help=False,
                                             formatter_class=CustomHelpFormatter,
                                             description='Create dereplicated mingle-compatible database.')

    req_args = create_db_parser.add_argument_group('required arguments')
    req_args.add_argument('genome_prot_dir', help="directory containing amino acid genes for each genome")
    req_args.add_argument('output_dir', help='directory to store results')

    optonal_args = create_db_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('-m', '--max_taxa', type=int, default=50, help='maximum taxa to retain in a named group')
    optonal_args.add_argument('-r', '--rank', type=int, default=5,
                              help='rank to preform dereplication [0=domain, 6=species]')
    optonal_args.add_argument('-p', '--per_identity', type=float, default=90.0,
                              help="percent identity for subsampling similar genes")
    optonal_args.add_argument('-a', '--per_aln_len', type=float, default=90.0,
                              help="percent alignment length for subsampling similar genes")
    optonal_args.add_argument('-x', '--extension', default='faa',
                              help="extension of files with called genes (other files in directory are ignored)")
    optonal_args.add_argument('--taxonomy', default=None, help='taxonomy string for each genome')
    optonal_args.add_argument('--type_strains', default=None,
                              help='file specifying type strains that should not be filtered')
    optonal_args.add_argument('--genomes_to_process', default=None,
                              help='list of genomes to retain instead of performing taxon subsampling')
    optonal_args.add_argument('--keep_all_genes', action="store_true", default=False,
                              help='restricts filtering to taxa')
    optonal_args.add_argument('--no_reformat_gene_ids', action="store_true", default=False,
                              help='do not reformat gene identifies')
    optonal_args.add_argument('--cpus', type=int, default=1, help='number of cpus to use')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    arb_db_parser = subparsers.add_parser('arb_db',
                                          add_help=False,
                                          formatter_class=CustomHelpFormatter,
                                          description='Create an ARB-compatible database of sequences.')
    req_args = arb_db_parser.add_argument_group('required arguments')
    req_args.add_argument('-a', '--alignment_file', help='MSA of proteins to be included in the database',
                          required=True)
    req_args.add_argument('-t', '--taxonomy_file', help='taxonomic assignment of each reference genomes', required=True)
    req_args.add_argument('-o', '--output_file', help='output file', required=True)
    optonal_args = arb_db_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # calculate Robinson-Foulds distance between trees
    robinson_foulds_parser = subparsers.add_parser('robinson_foulds',
                                                   add_help=False,
                                                   formatter_class=CustomHelpFormatter,
                                                   description='Calculate Robinson-Foulds distance between trees.')

    req_args = robinson_foulds_parser.add_argument_group('required arguments')
    req_args.add_argument('--tree1', help="first input tree")
    req_args.add_argument('--tree2', help='second input tree')

    optonal_args = robinson_foulds_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--weighted', help="perform weighted Robinson-Foulds", action='store_true')
    optonal_args.add_argument('--taxa_list', help='prune trees to specified by taxa before comparison')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # supported bipartitions of common taxa shared between two trees
    supported_splits_parser = subparsers.add_parser('supported_splits',
                                                    add_help=False,
                                                    formatter_class=CustomHelpFormatter,
                                                    description='Supported bipartitions of common taxa shared between two trees.')

    req_args = supported_splits_parser.add_argument_group('required arguments')
    req_args.add_argument('--tree1', help="first input tree")
    req_args.add_argument('--tree2', help='second input tree')

    optonal_args = supported_splits_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--split_file',
                              help='output file with information about congruent and incongruent splits')
    optonal_args.add_argument('-s', '--min_support', type=float, default=70, help='minimum support to consider split')
    optonal_args.add_argument('-d', '--max_depth', type=float, default=1.0,
                              help='ignore splits below the specified depth [0, 1.0]')
    optonal_args.add_argument('--taxa_list', help='prune trees to specified by taxa before comparison')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # supported bipartitions of common taxa shared between two trees
    missing_splits_parser = subparsers.add_parser('missing_splits',
                                                  add_help=False,
                                                  formatter_class=CustomHelpFormatter,
                                                  description='Report supported bipartitions in reference tree not in comparison tree.')

    req_args = missing_splits_parser.add_argument_group('required arguments')
    req_args.add_argument('--ref_tree', help="reference tree")
    req_args.add_argument('--compare_tree', help='comparison tree')

    optonal_args = missing_splits_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('-s', '--min_support', type=float, default=70,
                              help='minimum support in reference tree to consider split')
    optonal_args.add_argument('--taxa_list', help='prune trees to specified by taxa before comparison')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # reroot tree at midpoint
    midpoint_parser = subparsers.add_parser('midpoint',
                                            add_help=False,
                                            formatter_class=CustomHelpFormatter,
                                            description='Reroot tree at midpoint.')
    req_args = midpoint_parser.add_argument_group('required arguments')
    req_args.add_argument('--in_tree', help="tree to reroot")
    req_args.add_argument('--out_tree', help="output tree")

    optonal_args = midpoint_parser.add_argument_group('optional arguments')
    optonal_args.add_argument('--silent', help="suppress output", action='store_true')
    optonal_args.add_argument('-h', '--help', action="help", help="show help message")

    # get and check options
    options = None
    if (len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv == '--help'):
        print_help()
        sys.exit(0)
    else:
        options = parser.parse_args()

    if hasattr(options, 'output_dir'):
        logger_setup(options.output_dir, "genetreetk.log", "GeneTreeTk", version(), options.silent)
    else:
        logger_setup(None, "genetreetk.log", "GeneTreeTk", version(), options.silent)

    # do what we came here to do
    try:
        parser = OptionsParser()
        if (False):
            # import pstats
            # p = pstats.Stats('prof')
            # p.sort_stats('cumulative').print_stats(10)
            # p.sort_stats('time').print_stats(10)
            import cProfile
            cProfile.run('parser.parse_options(options)', 'prof')
        elif False:
            import pdb
            pdb.run(parser.parse_options(options))
        else:
            parser.parse_options(options)
    except SystemExit as se:
        print(repr(se))
        print("\nControlled exit resulting from an unrecoverable error or warning.")
    except:
        print("\nUnexpected error:", sys.exc_info()[0])
        raise


class OptionsParser:
    load_module_dependencies = False

    def __init__(self, path_to_dependency_config_file: str = None):
        """Initialization"""
        self.logger = logging.getLogger('timestamp')

    def blast(self, options):
        """Infer gene tree using BLAST."""
        
        check_file_exists(options.query_proteins)
        check_file_exists(options.db_file)
        check_file_exists(options.taxonomy_file)
        
        # sanity check arguments
        if options.prot_model == 'AUTO' and options.tree_program != 'raxml':
            self.logger.error("The 'AUTO' protein model can only be used with RAxML.")
            sys.exit(-1)

        blast_workflow = BlastWorkflow(options.cpus)
        blast_workflow.run(options.query_proteins,
                           options.db_file,
                           options.custom_db_file,
                           options.taxonomy_file,
                           options.custom_taxonomy_file,
                           options.evalue,
                           options.per_identity,
                           options.per_aln_len,
                           options.max_matches,
                           options.homology_search,
                           options.min_per_taxa,
                           options.consensus,
                           options.min_per_bp,
                           options.use_trimAl,
                           options.restrict_taxon,
                           options.msa_program,
                           options.tree_program,
                           options.prot_model,
                           options.skip_rooting,
                           options.output_dir)
                           
    def concat(self, options):
        """Infer concatenated gene tree."""
        
        make_sure_path_exists(options.output_dir)

        c = Concatenate(options.cpus)
        c.run(options.gene_dirs,
                options.min_per_gene,
                options.min_per_bps,
                options.tree_program,
                options.prot_model,
                options.split_chars,
                options.output_dir)
             
    def reduce(self, options):
        """Infer tree for reduced set of genes."""
        
        check_file_exists(options.homolog_file)
        check_file_exists(options.gene_ids)
        check_file_exists(options.taxonomy_file)
        
        make_sure_path_exists(options.output_dir)

        r = Reduce(options.cpus)
        r.run(options.homolog_file, 
                options.gene_ids, 
                options.taxonomy_file,
                options.min_per_taxa,
                options.consensus,
                options.min_per_bp,
                options.use_trimAl,
                options.msa_program,
                options.tree_program,
                options.prot_model,
                options.output_dir)
                
    def bootstrap(self, options):
        """Calculate bootstrap support for tree."""
        
        check_file_exists(options.tree)

        bootstrap = Bootstrap(options.cpus)
        bootstrap.run(options.tree, 
                    options.msa_file,
                    options.tree_program,
                    options.prot_model,
                    options.num_replicates,
                    options.output_dir)
                    
    def prune(self, options):
        """Prune tree."""
        
        check_file_exists(options.tree)
        check_file_exists(options.taxa_to_retain)

        prune = Prune()
        prune.run(options.tree,
                    options.taxa_to_retain,
                    options.output_tree)
    
    def prokka(self, options):
        """Run Prokka across multiple genome bins."""

        prokka = Prokka(options.cpus)
        prokka.run(options.genome_dir, 
                    options.kingdom, 
                    options.extension, 
                    options.output_dir)
                    
    def create_db(self, options):      
        """Create dereplicated GeneTreeTk-compatible database."""

        create_db = CreateDatabase(options.cpus)
        create_db.run(options.taxonomy,
                         options.type_strains,
                         options.genome_prot_dir,
                         options.extension,
                         options.max_taxa,
                         options.rank,
                         options.per_identity,
                         options.per_aln_len,
                         options.genomes_to_process,
                         options.keep_all_genes,
                         options.no_reformat_gene_ids,
                         options.output_dir)

    def create_arb_db(self, options):
        ArbDbCreator().create_from_protein_alignment(
            alignment_file=options.alignment_file,
            taxonomy_file=options.taxonomy_file,
            output_file=options.output_file)
                         
    def robinson_foulds(self, options):
        """Compare unrooted trees using common statistics."""
        
        check_file_exists(options.tree1)
        check_file_exists(options.tree2)

        tc = TreeCompare()
        if options.weighted:
            wrf = tc.weighted_robinson_foulds(options.tree1, 
                                                options.tree2,
                                                options.taxa_list)
            print(('Weighted Robinson-Foulds: %.3f' % wrf))
        else:
            rf, normalized_rf = tc.robinson_foulds(options.tree1, 
                                                    options.tree2,
                                                    options.taxa_list)
            print(('Robinson-Foulds: %d' % rf))
            print(('Normalized Robinson-Foulds: %.3f' % normalized_rf))
                         
    def supported_splits(self, options):
        """Supported bipartitions of common taxa shared between two trees."""
        
        check_file_exists(options.tree1)
        check_file_exists(options.tree2)
        
        tc = TreeCompare()
        tc.supported_splits(options.tree1, 
                            options.tree2,
                            options.split_file,
                            options.min_support,
                            options.max_depth,
                            options.taxa_list)
        
    def missing_splits(self, options):
        """Report supported bipartitions in reference tree not in comparison tree."""
        
        check_file_exists(options.ref_tree)
        check_file_exists(options.compare_tree)
        
        tc = TreeCompare()
        tc.report_missing_splits(options.ref_tree, 
                                    options.compare_tree,
                                    options.min_support,
                                    options.taxa_list)
                                    
    def midpoint(self, options):
        """"Midpoint root tree."""
        
        check_file_exists(options.in_tree)
        
        tree = dendropy.Tree.get_from_path(options.in_tree, 
                                            schema='newick', rooting='force-rooted', 
                                            preserve_underscores=True)
        tree.reroot_at_midpoint()
        
        tree.write_to_path(options.out_tree, 
                            schema='newick', 
                            suppress_rooting=True, 
                            unquoted_underscores=True)

    def orthologue(self, options):
        """Infer gene tree using BLAST after Orthologue clustering."""
        
        check_file_exists(options.query_proteins)
        check_file_exists(options.db_file)
        check_file_exists(options.taxonomy_file)

        # sanity check arguments
        if options.prot_model == 'AUTO' and options.tree_program != 'raxml':
            self.logger.error("The 'AUTO' protein model can only be used with RAxML.")
            sys.exit(-1)

        workflow = OrthologueWorkflow(options.cpus)
        workflow.run(
            query_proteins=options.query_proteins,
            db_file=options.db_file,
            #custom_db_file=options.custom_db_file,
            taxonomy_file=options.taxonomy_file,
            #custom_taxonomy_file=options.custom_taxonomy_file,
            evalue=options.evalue,
            per_identity=options.per_identity,
            per_aln_len=options.per_aln_len,
            max_matches=options.max_matches,
            homology_search=options.homology_search,
            min_per_taxa=options.min_per_taxa,
            consensus=options.consensus,
            min_per_bp=options.min_per_bp,
            use_trimAl=options.use_trimAl,
            restrict_taxon=options.restrict_taxon,
            msa_program=options.msa_program,
            tree_program=options.tree_program,
            prot_model=options.prot_model,
            skip_rooting=options.skip_rooting,
            output_dir=options.output_dir)



    def parse_options(self, options):
        """Parse user options and call the correct pipeline(s)"""

        if options.subparser_name == 'blast':
            self.blast(options)
        elif options.subparser_name == 'concat':
            self.concat(options)
        elif options.subparser_name == 'orthologue':
            self.orthologue(options)
        elif options.subparser_name == 'reduce':
            self.reduce(options)
        elif options.subparser_name == 'bootstrap':
            self.bootstrap(options)
        elif options.subparser_name == 'prune':
            self.prune(options)
        elif options.subparser_name == 'prokka':
            self.prokka(options)
        elif options.subparser_name == 'create_db':
            self.create_db(options)
        elif options.subparser_name == 'arb_db':
            self.create_arb_db(options)
        elif options.subparser_name == 'robinson_foulds':
            self.robinson_foulds(options)   
        elif options.subparser_name == 'supported_splits':
            self.supported_splits(options)
        elif options.subparser_name == 'missing_splits':
            self.missing_splits(options)
        elif options.subparser_name == 'midpoint':
            self.midpoint(options)   
        else:
            self.logger.error('  [Error] Unknown GeneTreeTk command: ' + options.subparser_name + '\n')
            sys.exit()

        return 0
