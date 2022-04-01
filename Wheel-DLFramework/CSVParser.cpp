#include "CSVParser.h"
namespace DataPreprocessing
{
	std::vector<std::vector<std::string>> DataPreprocessing::CSVParser::GetParsedResult()
	{
		std::ifstream input_file;
		input_file.open(m_file_path);

		std::string line = "";

		std::vector<std::vector<std::string>> data_parsed;

		while (std::getline(input_file, line))
		{
			std::string current_str;
			std::stringstream ss(line);
			std::vector <std::string> parsed_line;

			while (std::getline(ss, current_str, ','))
			{
				parsed_line.emplace_back(current_str);
			}

			data_parsed.emplace_back(parsed_line);
			line = "";
		}
		return data_parsed;
	}
}
