#pragma once
#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace DataPreprocessing
{
	class CSVParser
	{
	public:
		CSVParser(std::string file_path): m_file_path(file_path){}

		std::vector<std::vector<std::string>> GetParsedResult();

	private:
		void Parse();
		std::vector<std::vector<std::string>> m_parsed_result;
		std::string m_file_path;
	};
}
#endif
